import asyncio
import json
import statistics
import threading
import time
import traceback
from datetime import datetime

from . import board, datasets, doctor, engine, gpu, grader, lmx, metrics, notify, paths, schedule, speed, stack, store, suite, swap
from .profiles import Profile

RAM_FLOOR_GB = 4.0
GPU_WAIT_S = 1800
GPU_CHECK_S = 30
YIELD_STRIKES = 2
YIELD_RETRY_S = 600
MIN_WINDOW_S = 900
SPEED_WINDOW_S = 1800


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


class Halt:
    """Why a run has to stop early, set once by whichever guard notices first; the runners poll it."""

    def __init__(self):
        self.event = threading.Event()
        self.reason = None

    def set(self, reason):
        if not self.event.is_set():
            self.reason = reason
            self.event.set()

    def is_set(self):
        return self.event.is_set()


class MemoryGuard(threading.Thread):
    """Unloads the model and stops the run if free RAM drops under 4 GB, so a benchmark can never take the desktop down."""

    def __init__(self, halt, events):
        super().__init__(daemon=True)
        self.halt = halt
        self.events = events
        self.stopped = threading.Event()

    def run(self):
        while not self.stopped.wait(5):
            available = gpu.mem_available_gb()
            if available < RAM_FLOOR_GB:
                self.events.emit("abort", reason=f"free RAM {available:.1f} GB")
                swap.unload()
                self.halt.set("memory")
                return

    def stop(self):
        self.stopped.set()


class GpuWatch(threading.Thread):
    """For runs that give way: stops the run once another program has used the GPU for a minute (a game, ComfyUI)."""

    def __init__(self, halt, events):
        super().__init__(daemon=True)
        self.halt = halt
        self.events = events
        self.stopped = threading.Event()
        self.found = []

    def run(self):
        strikes = 0
        while not self.stopped.wait(GPU_CHECK_S):
            found = gpu.contention(samples=2)
            strikes = strikes + 1 if found else 0
            if strikes >= YIELD_STRIKES:
                self.found = found
                self.events.emit("yielding", to=gpu.describe_contention(found))
                self.halt.set("gpu")
                return

    def stop(self):
        self.stopped.set()


def wait_for_gpu(events):
    """Waits up to 30 minutes while other GPU work runs (a game, ComfyUI) so it can't skew speed; returns what was still
    running when it had to go ahead anyway."""
    deadline = time.time() + GPU_WAIT_S
    announced = False
    while True:
        found = gpu.contention()
        if not found:
            return []
        if time.time() > deadline:
            return [process["name"] for process in found]
        if not announced:
            events.emit("waiting", reason=f"GPU busy: {gpu.describe_contention(found)}")
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


def measure_speed(db, run_id, profile, spec, level, run_dir, events, halt, levels, context):
    """Measures speed once; a speed.json already in the run (resumed, or carried over with --reuse) is scored instead."""
    speed_file = run_dir / "speed.json"
    if speed_file.exists():
        result = json.loads(speed_file.read_text())
    else:
        result = asyncio.run(speed.SpeedRun(profile, spec, level, run_id, events.progress, halt, levels, context).run())
        if not halt.is_set():
            speed_file.write_text(json.dumps(result, indent=1))
    store.set_metrics(db, run_id, metrics.speed_metrics(result))
    return [process["name"] for process in result.get("contention", [])]


def add_energy(run_dir, task, watt_hours):
    """Adds one session's board energy for a task, so a run paused and continued over several nights sums them."""
    if watt_hours is None:
        return
    path = run_dir / "energy.json"
    energy = json.loads(path.read_text()) if path.exists() else {}
    energy[task] = round(energy.get(task, 0.0) + watt_hours, 3)
    path.write_text(json.dumps(energy, indent=1))


def park_for_window(db, run, events, window):
    """Ends this session before the window closes and frees the GPU; the run continues when the window next opens."""
    resumes = schedule.next_start(datetime.now(), window["start"]).timestamp()
    swap.unload()
    count = schedule.park(db, run["id"], resumes, f"the {window['start']}–{window['end']} window was closing", "paused")
    events.emit("paused", until=schedule.describe(resumes))
    if count == 1:
        notify.send("mbench paused", f"{run['model']}: stopped before {window['end']}; continues {schedule.describe(resumes)}",
                    run=run["id"], status="paused")


def give_way(db, run, events, found):
    """Frees the GPU for whatever else wants it and tries again in ten minutes, from the saved answers."""
    resumes = time.time() + YIELD_RETRY_S
    names = ", ".join(process["name"] for process in found) or "another program"
    swap.unload()
    count = schedule.park(db, run["id"], resumes, f"gave the GPU to {names}", "yielded")
    schedule.install_timer()
    events.emit("yielded", to=names, retry=schedule.describe(resumes))
    if count == 1:
        notify.send("mbench gave way", f"{run['model']} paused while {names} uses the GPU; it tries again every ten minutes",
                    run=run["id"], status="yielded")


def finished_note(db, run_id, model):
    entry = store.metrics_of(db, run_id).get("index.quality")
    return f"{model}: quality index {entry['value']:.1f}" if entry and entry.get("value") is not None else f"{model}: done"


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
    window = flags.get("window")
    halt = Halt()
    guard = MemoryGuard(halt, events)
    guard.start()
    watch = sampler = None
    store.update_run(db, run_id, status="running", started=time.time(), error=None)
    events.emit("started", model=profile.id, suite=run["suite"], effort=f"{run['effort']} ({level})")
    contention = []
    try:
        if suite.version_of(run["suite"]) != suite.VERSION:
            raise RuntimeError(f"this run was started under suite v{suite.version_of(run['suite'])} and this mbench runs "
                               f"v{suite.VERSION}; start a new run, with --reuse {run_id} to carry over what still applies")
        deadline = schedule.window_deadline(window, datetime.now())
        if deadline and deadline - time.time() < MIN_WINDOW_S:
            park_for_window(db, run, events, window)
            return
        if schedule.gives_way(flags):
            found = gpu.contention()
            if found:
                give_way(db, run, events, found)
                return
        else:
            contention += wait_for_gpu(events)
        seconds = swap.ensure_loaded(profile.id)
        info = swap.server_info(profile.id)
        (run_dir / "server_info.json").write_text(json.dumps(info, indent=1))
        capacity = swap.capacity(info)
        profile.context = profile.context or capacity["context"]
        context = swap.positive(profile.context, capacity["context"])
        hardware, build = gpu.describe(), stack.build(profile.engine, info)
        moved = stack.changed(store.last_complete(db, run["model"], run_id), hardware, build)
        if moved:
            flags["stack_change"] = moved
            events.emit("stack", changed="; ".join(moved))
        store.update_run(db, run_id, hardware=hardware, flags=flags, profile=profile.to_dict(), harness=stack.harness(),
                         server={"load_s": seconds, **swap.trimmed_info(info), "build": build, "capacity": capacity})
        events.emit("loaded", seconds=seconds, context=context, slots=capacity["slots"], pool=capacity["pool"],
                    build=stack.build_label(build))
        checks = asyncio.run(doctor.run(profile, level, context, capacity, moved))
        (run_dir / "doctor.json").write_text(json.dumps(checks, indent=1))
        for check in checks:
            if check["status"] != "ok":
                events.emit("doctor", check=check["check"], status=check["status"], detail=check["detail"])
        if doctor.failures(checks):
            raise RuntimeError("doctor: " + "; ".join(f"{check['check']}: {check['detail']}" for check in doctor.failures(checks)))
        if schedule.gives_way(flags):
            watch = GpuWatch(halt, events)
            watch.start()
        if "speed" in tasks:
            if deadline and not (run_dir / "speed.json").exists() and deadline - time.time() < SPEED_WINDOW_S:
                park_for_window(db, run, events, window)
                return
            slots = capacity["slots"] or suite.CONCURRENCY
            levels = [each for each in spec["speed"]["concurrency"] if each <= slots] or [1]
            contention += measure_speed(db, run_id, profile, spec["speed"], level, run_dir, events, halt, levels, context)
        sampler = gpu.Sampler(interval_ms=1000)
        runner = engine.QualityRunner(profile, run_dir, level, events.progress, halt, capacity["slots"], capacity["pool"],
                                      deadline)
        failed_items = 0
        for task in suite.QUALITY_TASKS:
            if task not in tasks:
                continue
            if halt.is_set() or runner.drained:
                break
            items = datasets.build(task, spec[task], context)
            started = time.time()
            failures = asyncio.run(runner.run(task, items))
            add_energy(run_dir, task, sampler.energy_wh(started, time.time()))
            if failures:
                failed_items += len(failures)
                events.emit(task, failed=len(failures), example=failures[0]["error"][:200])
            if halt.is_set() or runner.drained:
                break
            graded = None
            if task == "lcb":
                events.emit("grading", task="lcb")
                graded = grader.grade(run_dir)
            store.set_metrics(db, run_id, metrics.task_metrics(task, metrics.load(run_dir, task), graded))
        if halt.is_set():
            if halt.reason == "memory":
                raise RuntimeError(f"stopped because free RAM fell below {RAM_FLOOR_GB:.0f} GB")
            give_way(db, run, events, watch.found if watch else [])
            return
        if runner.drained:
            park_for_window(db, run, events, window)
            return
        store.set_metrics(db, run_id, metrics.quality_index(store.metrics_of(db, run_id), run_dir))
        store.set_metrics(db, run_id, metrics.energy_metrics(run_dir))
        if flags.get("submit"):
            which = flags["submit"] if flags["submit"] in lmx.SUBMIT_CHOICES else "all"
            submit(db, run_id, profile, level, run_dir, events, which)
        flags = {**((store.get_run(db, run_id) or {}).get("flags") or flags),
                 "contended": sorted(set(contention)), "failed_items": failed_items}
        flags.pop("parked", None)
        store.update_run(db, run_id, status="complete", finished=time.time(), flags=flags)
        events.emit("complete")
        notify.send("mbench finished", finished_note(db, run_id, profile.id), run=run_id, status="complete")
    except Exception as error:
        store.update_run(db, run_id, status="failed", finished=time.time(), error=str(error)[:500])
        events.emit("failed", error=str(error)[:300])
        events.log(traceback.format_exc())
        notify.send("mbench failed", f"{profile.id}: {str(error)[:200]}", run=run_id, status="failed")
    finally:
        guard.stop()
        if watch:
            watch.stop()
        if sampler:
            sampler.close()
        try:
            board.build()
        except Exception as error:
            events.log(f"leaderboard rebuild failed: {error!r}")
        try:
            schedule.tick(db)
        except Exception as error:
            events.log(f"starting the next scheduled run failed: {error!r}")
