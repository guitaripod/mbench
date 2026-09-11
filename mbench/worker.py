import asyncio
import json
import statistics
import threading
import time
import traceback

from . import board, datasets, engine, gpu, grader, lmx, metrics, paths, speed, store, suite, swap
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


def execute(run_id):
    db = store.connect()
    run = store.get_run(db, run_id)
    run_dir = paths.RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events = Events(run_dir)
    profile = Profile(**run["profile"])
    spec = suite.SUITES[run["suite"].split("/")[0]]
    flags = run["flags"] or {}
    tasks = flags.get("tasks") or ["speed", *suite.QUALITY_TASKS]
    abort = threading.Event()
    guard = MemoryGuard(abort, events)
    guard.start()
    store.update_run(db, run_id, status="running", started=time.time(), error=None)
    events.emit("started", model=profile.id, suite=run["suite"])
    contention = []
    try:
        contention += [app["name"] for app in wait_for_gpu(events)]
        seconds = swap.ensure_loaded(profile.id)
        info = swap.server_info(profile.id)
        (run_dir / "server_info.json").write_text(json.dumps(info, indent=1))
        profile.context = profile.context or swap.context_from(info)
        store.update_run(db, run_id, server={"load_s": seconds, **swap.trimmed_info(info)}, profile=profile.to_dict())
        events.emit("loaded", seconds=seconds, context=profile.context)
        speed_file = run_dir / "speed.json"
        if "speed" in tasks and not speed_file.exists():
            result = asyncio.run(speed.SpeedRun(profile, spec["speed"], run["effort"], run_id, events.progress, abort).run())
            if not abort.is_set():
                speed_file.write_text(json.dumps(result, indent=1))
            store.set_metrics(db, run_id, metrics.speed_metrics(result))
            contention += [app["name"] for app in result.get("contention", [])]
        runner = engine.QualityRunner(profile, run_dir, run["effort"], suite.CONCURRENCY, events.progress, abort)
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
        store.set_metrics(db, run_id, metrics.quality_index(store.metrics_of(db, run_id)))
        if flags.get("submit"):
            which = flags["submit"] if flags["submit"] in lmx.SUBMIT_CHOICES else "all"
            submit(db, run_id, profile, run["effort"], run_dir, events, which)
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
