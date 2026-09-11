import asyncio
import json
import statistics
import threading
import time
import traceback

from . import board, datasets, engine, gpu, grader, lmx, metrics, paths, schedule, speed, store, suite, swap
from .profiles import Profile

RAM_FLOOR_GB = 4.0
GPU_WAIT_S = 1800
GPU_BUSY_PERCENT = 20


class Events:
    """Progress for `mbench run` and `mbench status`: one JSON line per update plus a readable worker.log."""

    def __init__(self, run_dir):
        self.events = run_dir / "events.jsonl"
        self.log_path = run_dir / "worker.log"
        self.last = (None, None)

    def emit(self, phase, **fields):
        entry = {"t": round(time.time(), 1), "phase": phase, **fields}
        with self.events.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        self.log(f"{phase} " + " ".join(f"{key}={value}" for key, value in fields.items()))

    def log(self, message):
        with self.log_path.open("a") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")

    def progress(self, phase, done, total):
        """Emits the first and last item and every 2% between, so the event file stays small on long tasks."""
        step = max(1, total // 50)
        if (phase, done) == self.last:
            return
        if done in (0, total) or done % step == 0 or phase != self.last[0]:
            self.last = (phase, done)
            self.emit(phase, done=done, total=total)


class MemoryGuard(threading.Thread):
    """Unloads the model and stops the run if free RAM drops under 4 GB, so a benchmark can never take the desktop down."""

    def __init__(self, abort, events):
        super().__init__(daemon=True)
        self.abort = abort
        self.events = events
        self.stopped = threading.Event()

    def run(self):
        while not self.stopped.wait(5):
            available = gpu.mem_available_gb()
            if available < RAM_FLOOR_GB:
                self.events.emit("abort", reason=f"free RAM {available:.1f} GB")
                swap.unload()
                self.abort.set()
                return

    def stop(self):
        self.stopped.set()


def wait_for_gpu(events):
    """Waits up to 30 minutes while something else keeps the GPU busy (a generation, a game, another chat) so it can't skew speed."""
    deadline = time.time() + GPU_WAIT_S
    announced = False
    while True:
        load = gpu.utilization()
        if load < GPU_BUSY_PERCENT:
            return []
        if time.time() > deadline:
            return [f"GPU {load:.0f}% busy at start"]
        if not announced:
            holders = ", ".join(f"{app['name']} {app['mib'] // 1024} GB" for app in gpu.heavy_apps())
            events.emit("waiting", reason=f"GPU {load:.0f}% busy" + (f" ({holders})" if holders else ""))
            announced = True
        time.sleep(30)


def submit(db, run_id, profile, effort, run_dir, events, which):
    if not lmx.available():
        events.emit("localmaxxing", skipped="lmx is not installed or not logged in")
        return
    missing = lmx.missing_fields(profile)
    if missing:
        events.emit("localmaxxing", skipped=f"set {', '.join(missing)} for {profile.id} in {paths.PROFILES}")
        return
    if which in ("all", "speed"):
        events.emit("localmaxxing", step="speed runs")
        for entry in lmx.speed_runs(profile, run_dir, run_id, events.log):
            store.add_submission(db, run_id, entry["kind"], entry["remote_id"], entry["value"], entry)
    if which in ("all", "evals"):
        events.emit("localmaxxing", step="GSM8K and HellaSwag shards")
        extra = {}
        for dataset, entries in lmx.shard_runs(profile, effort, run_dir, events.log).items():
            for entry in entries:
                store.add_submission(db, run_id, f"shard.{dataset}", str(entry["shard"]), entry["accuracy"], entry)
            if entries:
                extra[f"lmx.{dataset}"] = {"value": statistics.mean(entry["accuracy"] for entry in entries),
                                           "unit": "%", "n": len(entries)}
        store.set_metrics(db, run_id, extra)


def measure_speed(db, run_id, profile, spec, level, run_dir, events, abort):
    """Measures speed once; a speed.json already in the run (resumed, or carried over with --reuse) is scored instead."""
    speed_file = run_dir / "speed.json"
    if speed_file.exists():
        result = json.loads(speed_file.read_text())
    else:
        result = asyncio.run(speed.SpeedRun(profile, spec, level, run_id, events.progress, abort).run())
        if not abort.is_set():
            speed_file.write_text(json.dumps(result, indent=1))
    store.set_metrics(db, run_id, metrics.speed_metrics(result))
    return [app["name"] for app in result.get("contention", [])]


def execute(run_id):
    db = store.connect()
    run = store.get_run(db, run_id)
    run_dir = paths.RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events = Events(run_dir)
    profile = Profile(**run["profile"])
    spec = suite.SUITES[suite.kind_of(run["suite"])]
    flags = run["flags"] or {}
    tasks = flags.get("tasks") or ["speed", *suite.QUALITY_TASKS]
    level = flags.get("effort_level") or run["effort"]
    abort = threading.Event()
    guard = MemoryGuard(abort, events)
    guard.start()
    store.update_run(db, run_id, status="running", started=time.time(), error=None)
    events.emit("started", model=profile.id, suite=run["suite"], effort=f"{run['effort']} ({level})")
    contention = []
    try:
        if suite.version_of(run["suite"]) != suite.VERSION:
            raise RuntimeError(f"this run was started under suite v{suite.version_of(run['suite'])} and this mbench runs "
                               f"v{suite.VERSION}; start a new run, with --reuse {run_id} to carry over what still applies")
        contention += wait_for_gpu(events)
        seconds = swap.ensure_loaded(profile.id)
        info = swap.server_info(profile.id)
        (run_dir / "server_info.json").write_text(json.dumps(info, indent=1))
        profile.context = profile.context or swap.context_from(info)
        store.update_run(db, run_id, server={"load_s": seconds, **swap.trimmed_info(info)}, profile=profile.to_dict())
        events.emit("loaded", seconds=seconds, context=profile.context)
        if "speed" in tasks:
            contention += measure_speed(db, run_id, profile, spec["speed"], level, run_dir, events, abort)
        runner = engine.QualityRunner(profile, run_dir, level, suite.CONCURRENCY, events.progress, abort)
        failed_items = 0
        for task in suite.QUALITY_TASKS:
            if task not in tasks or abort.is_set():
                continue
            items = datasets.build(task, spec[task], profile.context)
            failures = asyncio.run(runner.run(task, items))
            if failures:
                failed_items += len(failures)
                events.emit(task, failed=len(failures), example=failures[0]["error"][:200])
            graded = None
            if task == "lcb" and not abort.is_set():
                events.emit("grading", task="lcb")
                graded = grader.grade(run_dir)
            store.set_metrics(db, run_id, metrics.task_metrics(task, metrics.load(run_dir, task), graded))
        if abort.is_set():
            raise RuntimeError(f"stopped because free RAM fell below {RAM_FLOOR_GB:.0f} GB")
        store.set_metrics(db, run_id, metrics.quality_index(store.metrics_of(db, run_id), run_dir))
        if flags.get("submit"):
            which = flags["submit"] if flags["submit"] in lmx.SUBMIT_CHOICES else "all"
            submit(db, run_id, profile, level, run_dir, events, which)
        flags.update(contended=sorted(set(contention)), failed_items=failed_items)
        store.update_run(db, run_id, status="complete", finished=time.time(), flags=flags)
        events.emit("complete")
    except Exception as error:
        store.update_run(db, run_id, status="failed", finished=time.time(), error=str(error)[:500])
        events.emit("failed", error=str(error)[:300])
        events.log(traceback.format_exc())
    finally:
        guard.stop()
        try:
            board.build()
        except Exception as error:
            events.log(f"leaderboard rebuild failed: {error!r}")
        try:
            schedule.tick(db)
        except Exception as error:
            events.log(f"starting the next scheduled run failed: {error!r}")
