import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NoReturn

from . import __version__, board, doctor, hosts, lmx, metrics, paths, phone, profiles, schedule, sources, stack, store, suite, swap
from .engine import resolve_effort
from .units import ACTIVE, busy, reconcile, spawn, unit_active, unit_name

DURATIONS = {"full": "2–5 hours", "quick": "45–90 minutes", "phone": "30–60 minutes", "smoke": "about 10 minutes"}
REUSE_FILES = {"speed": ("speed.json",), "lcb": ("lcb.jsonl", "lcb.graded.jsonl")}
REUSE_STATUSES = ("complete", "failed", "cancelled")


def fail(message) -> NoReturn:
    print(f"mbench: {message}", file=sys.stderr)
    sys.exit(1)


def resolve_profile(model):
    if not paths.LLAMA_SWAP_CONFIG.exists():
        fail(f"no llama-swap config at {paths.LLAMA_SWAP_CONFIG}; point MBENCH_SWAP_CONFIG at yours")
    try:
        return profiles.resolve(model)
    except KeyError as error:
        fail(str(error.args[0]))


def duration(seconds):
    if seconds is None or seconds < 0 or seconds > 86400 * 3:
        return "–"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


class Follower:
    """Prints a run's events; on a terminal, progress for the current phase rewrites one line instead of scrolling."""

    def __init__(self, run_id):
        self.path = paths.RUNS / run_id / "events.jsonl"
        self.position = 0
        self.phase_start = {}
        self.tty = sys.stdout.isatty()
        self.open_line = False

    def poll(self):
        if not self.path.exists():
            return
        with self.path.open() as handle:
            handle.seek(self.position)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line.endswith("\n"):
                    handle.seek(start)
                    break
                self.render(json.loads(line))
            self.position = handle.tell()

    def render(self, event):
        phase = event["phase"]
        stamp = time.strftime("%H:%M", time.localtime(event["t"]))
        if "done" in event:
            first = self.phase_start.setdefault(phase, (event["t"], event["done"]))
            rate = (event["done"] - first[1]) / max(1e-6, event["t"] - first[0])
            eta = (event["total"] - event["done"]) / rate if rate > 0 else None
            filled = int(24 * event["done"] / max(1, event["total"]))
            text = f"{stamp}  {phase:<10} [{'#' * filled}{'.' * (24 - filled)}] {event['done']}/{event['total']}  eta {duration(eta)}"
            if self.tty:
                print("\r" + text.ljust(80), end="", flush=True)
                self.open_line = event["done"] < event["total"]
                if not self.open_line:
                    print()
            else:
                print(text)
            return
        if self.open_line:
            print()
            self.open_line = False
        details = " ".join(f"{key}={value}" for key, value in event.items() if key not in ("t", "phase"))
        print(f"{stamp}  {phase:<10} {details}".rstrip())


def definition_of(run):
    return suite.DEFINITIONS.get(suite.version_of(run["suite"]), suite.DEFINITIONS[suite.VERSION])


def summary(db, run_id):
    metrics_ = store.metrics_of(db, run_id)
    definition = definition_of(store.get_run(db, run_id))

    def show(key, label, unit=""):
        entry = metrics_.get(key)
        if entry and entry.get("value") is not None:
            interval = f"  ({entry['lo']:.1f}–{entry['hi']:.1f})" if entry.get("lo") is not None and unit == "%" else ""
            print(f"  {label:<26} {entry['value']:>8.1f} {unit}{interval}")

    show("index.quality", "Quality index", "%")
    for task in definition["index_tasks"]:
        show(f"{task}.score", definition["labels"][task], "%")
    show("speed.decode", "Decode, one request", "tok/s")
    show("speed.peak", "Peak throughput", "tok/s")
    show("energy.per_correct", "Energy per correct answer", "Wh")
    show("speed.ttft.32000", "First token, 32k prompt", "s")


def follow(run_id):
    db = store.connect()
    follower = Follower(run_id)
    print("Following progress. Ctrl-C detaches; the run keeps going.")
    try:
        while True:
            follower.poll()
            run = store.get_run(db, run_id)
            if run["status"] not in ACTIVE:
                follower.poll()
                if follower.open_line:
                    print()
                print(f"{run_id}: {run['status']}" + (f" ({run['error']})" if run.get("error") else ""))
                if run["status"] == "complete":
                    summary(db, run_id)
                    print(f"Leaderboard: {paths.BOARD}")
                return
            if not unit_active(run_id) and time.time() - run["created"] > 60:
                reconcile(db)
            time.sleep(2)
    except KeyboardInterrupt:
        print(f"\nDetached. {run_id} keeps running; see `mbench status` or `mbench logs -f`.")


def level_of(run):
    return (run.get("flags") or {}).get("effort_level") or run.get("effort")


def reuse_problem(run, profile, level, suite_name):
    """Why an earlier run's answers can't stand in for this one's, or None when they can."""
    if run["model"] != profile.id:
        return f"it measured {run['model']}"
    if run["status"] not in REUSE_STATUSES:
        return f"it is {run['status']}"
    if suite.kind_of(run["suite"]) != suite_name:
        return f"it ran the {suite.kind_of(run['suite'])} suite, not {suite_name}"
    if level_of(run) != level:
        return f"it ran at effort {level_of(run)}, not {level}"
    if run.get("fingerprint") != profile.fingerprint:
        return "the model's llama-swap command or launcher changed since"
    return None


def task_files(task):
    return REUSE_FILES.get(task, (f"{task}.jsonl",))


def carried_tasks(run, tasks):
    """The tasks whose answers the earlier run has and whose items and scoring are unchanged in this suite version."""
    source = paths.RUNS / run["id"]
    return [task for task in tasks if suite.reusable(task, suite.version_of(run["suite"]))
            and (source / task_files(task)[0]).exists()]


def reuse_source(db, requested, profile, level, suite_name, tasks):
    if requested != "latest":
        run = store.get_run(db, requested) or fail(f"no run {requested}")
        problem = reuse_problem(run, profile, level, suite_name)
        if problem:
            fail(f"can't reuse {requested}: {problem}")
        return run
    for run in store.list_runs(db, model=profile.id):
        if reuse_problem(run, profile, level, suite_name) is None and carried_tasks(run, tasks):
            return run
    return None


def carry_over(run, run_id, tasks):
    carried = carried_tasks(run, tasks)
    for task in carried:
        for name in task_files(task):
            source = paths.RUNS / run["id"] / name
            if source.exists():
                shutil.copyfile(source, paths.RUNS / run_id / name)
    return carried


def window_of(args):
    """The --at/--until schedule as (first start or None, daily window), or (None, None) to start now."""
    if args.until and not args.at:
        fail("--until needs --at: the window is --at to --until, every day until the runs finish")
    if not args.at:
        return None, None
    try:
        at, start = schedule.parse_at(args.at)
        end = schedule.parse_clock(args.until) if args.until else None
    except ValueError as error:
        fail(str(error))
    if end == start:
        fail("--at and --until are the same time")
    return at, {"start": start, "end": end}


def selected_tasks(args):
    tasks = ["speed", *suite.QUALITY_TASKS]
    chosen = set(args.only.split(",")) if args.only else set(tasks)
    skipped = set(args.skip.split(",")) if args.skip else set()
    unknown = (chosen | skipped) - set(tasks)
    if unknown:
        fail(f"unknown task {', '.join(sorted(unknown))}; tasks are {', '.join(tasks)}")
    return [task for task in tasks if task in chosen and task not in skipped]


def phone_run(chosen):
    """Phone models and desktop models can't share a run: they answer on different servers and are measured under
    different suites."""
    phones = [profile for profile in chosen if profile.phone]
    if phones and len(phones) != len(chosen):
        fail("a phone model and a desktop model can't be queued together; run them separately")
    return bool(phones)


def cmd_run(args):
    models = list(dict.fromkeys(args.model))
    if len(models) > 1 and args.reuse not in (None, "latest"):
        fail("--reuse with a run id works for one model; with several, --reuse picks each model's newest run")
    at, window = window_of(args)
    chosen = [resolve_profile(model) for model in models]
    on_phone = phone_run(chosen)
    for profile in chosen:
        host = hosts.for_profile(profile)
        if not host.reachable():
            fail(f"{'mbenchd' if on_phone else 'llama-swap'} is not answering at {host.base_url()}")
    tasks = selected_tasks(args)
    if on_phone:
        asked = [task for task in tasks if task not in suite.PHONE_TASKS]
        if asked and args.only:
            fail(f"a phone measures {', '.join(suite.PHONE_TASKS)}; {', '.join(asked)} would be scored on a smaller "
                 "sample than every other row. Score the same .gguf on the desktop and name that run with --quality-from")
        tasks = [task for task in tasks if task in suite.PHONE_TASKS]
    levels = {}
    for profile in chosen:
        try:
            levels[profile.id] = resolve_effort(profile, args.effort)
        except ValueError as error:
            fail(str(error))
        if args.submit and lmx.missing_fields(profile):
            fail(f"--submit needs {', '.join(lmx.missing_fields(profile))} for {profile.id} in {paths.PROFILES}")
    db = store.connect()
    reconcile(db)
    active = busy(store.list_runs(db), "phone" if on_phone else "gpu")
    suite_name = "smoke" if args.smoke else "phone" if on_phone else "quick" if args.quick else "full"
    now = datetime.now()
    begins = schedule.first_start(now, window, at).timestamp() if window else now.timestamp()
    immediate = None
    for position, profile in enumerate(chosen):
        level = levels[profile.id]
        source = reuse_source(db, args.reuse, profile, level, suite_name, tasks) if args.reuse else None
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{profile.id}"
        (paths.RUNS / run_id).mkdir(parents=True, exist_ok=True)
        carried = carry_over(source, run_id, tasks) if source else []
        flags = {"tasks": tasks, "submit": args.submit, "effort_level": level, "yield": not args.keep_gpu}
        if args.quality_from or profile.twin:
            flags["quality_from"] = args.quality_from or {"twin": profile.twin}
        if carried:
            flags["reused"] = {"run": source["id"], "tasks": carried}
        starts_now = window is None and not active and position == 0
        if not starts_now:
            flags.update(not_before=begins, window=window)
        store.insert_run(db, {
            "id": run_id, "model": profile.id, "name": profile.name, "suite": suite.label(suite_name),
            "effort": args.effort, "status": "queued" if starts_now else "scheduled", "harness": stack.harness(),
            "fingerprint": profile.fingerprint, "profile": profile.to_dict(),
            "hardware": hosts.for_profile(profile).describe(), "flags": flags,
            "note": args.note,
        })
        if position:
            when = "after the one before it"
        elif window:
            when = f"at {schedule.describe(begins)}"
        else:
            when = f"after {active[0]['id']}" if active else "now"
        print(f"{run_id}: {suite.label(suite_name)} on {profile.name} at effort {args.effort}"
              + (f" ({level})" if level != args.effort else "") + f", usually {DURATIONS[suite_name]}, starts {when}.")
        if args.reuse:
            print(f"  Carrying over {', '.join(carried)} from {source['id']}." if carried
                  else "  Nothing earlier still applies; measuring everything.")
        immediate = immediate or (run_id if starts_now else None)
    if window and window["end"]:
        print(f"Runs only between {window['start']} and {window['end']}; whatever is unfinished at {window['end']} "
              f"stops, frees the GPU and continues at {window['start']} the next day.")
    if len(chosen) > 1 or not immediate:
        schedule.install_timer()
    if not immediate:
        print("`mbench status` lists the schedule; `mbench cancel <run>` takes a run off it.")
        return
    others = [] if on_phone else [entry["model"] for entry in swap.running() if entry["model"] != chosen[0].id]
    if others:
        print(f"llama-swap will unload {', '.join(others)} to make room.")
    spawn(immediate, args.foreground)
    if not args.foreground and not args.detach and len(chosen) == 1:
        follow(immediate)


def cmd_phone(args):
    device = phone.Device()
    if args.action == "forward":
        processes = [phone.forward(18080, phone.SERVER_PORT), phone.forward(18081, phone.CONTROL_PORT)]
        print(f"Forwarding {phone.SERVER_URL} → llama-server and {phone.CONTROL_URL} → mbenchd. Ctrl-C to stop.")
        try:
            while all(process.poll() is None for process in processes):
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            for process in processes:
                process.terminate()
        return
    if args.action == "launch":
        print(f"mbenchd running as pid {phone.launch()}")
        for _ in range(30):
            if device.reachable():
                print(f"Answering at {device.control_url}.")
                return
            time.sleep(1)
        fail("the app launched but never answered; `mbench phone logs` may say why")
    if args.action == "push":
        for path in args.files:
            source = Path(path)
            if not source.exists():
                fail(f"no file {source}")
            print(f"Pushing {source.name} ({source.stat().st_size / 2**30:.2f} GB) to the phone…")
            print("  " + phone.push(source))
        return
    if args.action == "logs":
        target = Path(args.into or ".")
        target.mkdir(parents=True, exist_ok=True)
        health = device.health() or {}
        app = health.get("app") or {}
        for remote in filter(None, [app.get("log_path"), app.get("server_log_path")]):
            name = remote.rsplit("/", 1)[-1]
            local = target / name
            relative = remote.split("/Data/Application/", 1)[-1].split("/", 1)[-1] if "/Data/Application/" in remote else remote
            try:
                print(phone.pull(relative, local))
            except RuntimeError as error:
                print(f"  {name}: {error}")
        return
    health = device.health()
    if not health:
        fail(f"mbenchd is not answering at {device.control_url}; `mbench phone forward` and open the app")
    telemetry, server = health["telemetry"], health["server"]
    described = device.describe()
    print(f"{described['device']} · {described['soc'] or '?'} · {described['os']} · mbenchd {described['app']} · "
          f"llama.cpp {device.build()['version']}")
    print(f"  thermal   {telemetry['thermal_state']} for {telemetry['thermal_since']:.0f} s")
    print(f"  memory    {telemetry['footprint_mib']:.0f} MiB held, {telemetry['available_mib']:.0f} MiB still available")
    print(f"  battery   {telemetry['battery_level'] * 100:.0f}% {telemetry['battery_state']}")
    print(f"  server    {server['state']}" + (f" · {server['model']}" if server.get("model") else ""))
    for entry in device.models():
        print(f"  model     {entry['id']} ({entry['bytes'] / 2**30:.2f} GB)")


def latest_run(db, run_id=None):
    if run_id:
        return store.get_run(db, run_id) or fail(f"no run {run_id}")
    runs = store.list_runs(db)
    return runs[0] if runs else fail("no runs yet; start one with `mbench run <llama-swap model>`")


def print_schedule(db):
    waiting = sorted((run for run in store.list_runs(db) if run["status"] == "scheduled"),
                     key=lambda run: ((run.get("flags") or {}).get("not_before") or 0, run.get("created") or 0))
    for run in waiting:
        flags = run.get("flags") or {}
        window = flags.get("window") or {}
        start = f"starts {schedule.describe(flags['not_before'])}" if flags.get("not_before", 0) > time.time() else "due next"
        print(f"{run['id']}  {run['suite']}  scheduled, {start}"
              + (f", only {window['start']}–{window['end']}, continuing the next night if unfinished"
                 if window.get("end") else "")
              + (f" ({flags['parked']}; continues where it stopped)" if flags.get("parked") else ""))


def cmd_status(_args):
    db = store.connect()
    reconcile(db)
    active = [run for run in store.list_runs(db) if run["status"] in ACTIVE]
    if not active:
        print("No run in progress.")
        runs = [run for run in store.list_runs(db) if run["status"] != "scheduled"]
        if runs:
            last = runs[0]
            print(f"Last: {last['id']} {last['status']}" + (f" ({last['error']})" if last.get("error") else ""))
        print_schedule(db)
        return
    for run in active:
        events = paths.RUNS / run["id"] / "events.jsonl"
        last = json.loads(events.read_text().splitlines()[-1]) if events.exists() and events.read_text().strip() else {}
        detail = f"{last.get('phase', 'queued')} {last.get('done', '')}/{last.get('total', '')}".rstrip("/ ")
        print(f"{run['id']}  {run['suite']}  {detail}  running for {duration(time.time() - (run['started'] or run['created']))}")
    print_schedule(db)


def cmd_logs(args):
    db = store.connect()
    run = latest_run(db, args.run)
    path = paths.RUNS / run["id"] / "worker.log"
    if not path.exists():
        fail(f"{run['id']} has no log yet")
    if not args.follow:
        print(path.read_text(), end="")
        return
    with path.open() as handle:
        print(handle.read(), end="")
        try:
            while True:
                line = handle.readline()
                if line:
                    print(line, end="", flush=True)
                elif store.get_run(db, run["id"])["status"] not in ACTIVE:
                    return
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            return


def cmd_cancel(args):
    db = store.connect()
    run = latest_run(db, args.run)
    subprocess.run(["systemctl", "--user", "stop", unit_name(run["id"])], capture_output=True)
    store.update_run(db, run["id"], status="cancelled", finished=time.time())
    print(f"{run['id']}: cancelled. `mbench resume {run['id']}` picks it up where it stopped.")
    waiting = [other for other in store.list_runs(db) if other["status"] == "scheduled"]
    if waiting:
        print(f"{len(waiting)} more scheduled; `mbench status` lists them and `mbench cancel <run>` removes one.")
    else:
        schedule.remove_timer()


def cmd_resume(args):
    db = store.connect()
    reconcile(db)
    run = latest_run(db, args.run)
    if run["status"] in ACTIVE:
        fail(f"{run['id']} is already running")
    if busy(store.list_runs(db), (run.get("hardware") or {}).get("class") or "gpu") and not args.at:
        fail("another run is using the same device; add --at to schedule this one")
    if run["status"] == "complete":
        fail(f"{run['id']} already finished")
    if suite.version_of(run["suite"]) != suite.VERSION:
        fail(f"{run['id']} was measured under suite v{suite.version_of(run['suite'])} and this mbench runs v{suite.VERSION}; "
             f"`mbench run {run['model']} --reuse {run['id']}` starts a v{suite.VERSION} run that keeps what still applies")
    at, window = window_of(args)
    flags = {**(run.get("flags") or {}), "yield": not args.keep_gpu}
    if window:
        begins = schedule.first_start(datetime.now(), window, at).timestamp()
        store.update_run(db, run["id"], status="scheduled", error=None,
                         flags={**flags, "not_before": begins, "window": window})
        schedule.install_timer()
        print(f"{run['id']}: continues {schedule.describe(begins)}.")
        return
    store.update_run(db, run["id"], status="queued", error=None, flags=flags)
    spawn(run["id"], args.foreground)
    if not args.foreground and not args.detach:
        follow(run["id"])


def cell(entry, digits=1):
    if not entry or entry.get("value") is None:
        return "–"
    return f"{entry['value']:.{digits}f}"


def ranking_table(view, effort):
    """The ranked table as a header plus rows of strings, shared by the terminal and the markdown output."""
    header = ["Model", "Quality", *[view["taskShort"][task] for task in view["indexTasks"]],
              "tok/s", "Peak", "TTFT 32k", "Wh/correct", "Run"]
    rows = []
    for model in sorted(view["rankings"][effort], key=lambda model: -((model["index"] or {}).get("value") or -1)):
        speed = model["speed"]
        kind = model["run"]["kind"]
        rows.append([
            model["name"], cell(model["index"]), *[cell(model["tasks"].get(task)) for task in view["indexTasks"]],
            cell(speed.get("decode"), 0), cell(speed.get("peak"), 0), cell(speed.get("ttft.32000")),
            cell(model["energy"].get("perCorrect"), 2),
            datetime.fromtimestamp(model["run"]["finished"]).strftime("%d %b") + ("" if kind == "full" else f" {kind}"),
        ])
    return header, rows


def print_markdown(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] + ["--:"] * (len(header) - 1)) + "|")
    for row in rows:
        print("| " + " | ".join(str(value) for value in row) + " |")


def print_columns(header, rows):
    widths = [max(len(str(row[index])) for row in [header, *rows]) for index in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(str(value).ljust(width) if index == 0 else str(value).rjust(width)
                        for index, (value, width) in enumerate(zip(row, widths))))


def setups_note(setups):
    """Quality compares across cards; the speed columns only compare within one, so a mixed ranking says which is which."""
    detail = ", ".join(f"{entry['label']} ({entry['models']})" for entry in setups)
    return f"tok/s, Peak and Wh/correct come from {len(setups)} setups — {detail} — and only compare within one"


def cmd_ls(args):
    data = board.collect(store.connect())
    version = args.suite or suite.VERSION
    if version not in data["suites"]:
        fail(f"no suite v{version}; there are {', '.join('v' + name for name in data['suites'])}")
    view = data["suites"][version]
    if not view["rankings"].get(args.effort):
        others = ", ".join(effort for effort in view["efforts"] if effort != args.effort)
        older = [name for name, other in data["suites"].items() if name != version and other["efforts"]]
        print(f"No finished {args.effort}-effort runs on suite v{version} yet."
              + (f" There are {others}-effort runs: `mbench ls --effort <level>`." if others else "")
              + (f" Older suites have runs: `mbench ls --suite {older[0]}`." if older else "")
              + ("" if others or older else " Start one with `mbench run <llama-swap model>`."))
        return
    models = view["rankings"][args.effort]
    header, rows = ranking_table(view, args.effort)
    (print_markdown if args.markdown else print_columns)(header, rows)
    setups = view["hosts"].get(args.effort) or []
    if args.markdown:
        newest = max((model["run"]["finished"] or 0) for model in models)
        where = setups[0]["label"].split(" · ")[0] if len(setups) == 1 else f"{len(setups)} setups"
        print(f"\n_{len(rows)} models on {where}, suite {data['suite']}, {args.effort} effort, "
              f"newest run {datetime.fromtimestamp(newest):%d %b %Y}._"
              + (f"\n\n_{setups_note(setups)}_" if len(setups) > 1 else ""))
        return
    if len(setups) > 1:
        print(f"note: {setups_note(setups)}")
    for task, why in view["health"].get(args.effort, {}).items():
        print(f"note: {view['taskLabels'][task]} isn't separating these models ({why})")


def cmd_export(args):
    for path in board.export(store.connect(), Path(args.out or paths.DATA / "site")):
        print(path)


def comparison_run(db, name, effort):
    """A run id as given, or a model's headline run at the effort on the current suite."""
    run = store.get_run(db, name)
    if run:
        return run
    runs = [run for run in store.list_runs(db, model=name, status="complete")
            if suite.version_of(run["suite"]) == suite.VERSION]
    chosen = board.headline(runs, effort).get(name)
    return chosen or fail(f"{name} is neither a run id nor a model with a finished {effort}-effort v{suite.VERSION} run")


def verdict(entry):
    if entry["lo"] > 0:
        return "A ahead"
    if entry["hi"] < 0:
        return "B ahead"
    return "no clear difference"


def cmd_compare(args):
    db = store.connect()
    first, second = comparison_run(db, args.a, args.effort), comparison_run(db, args.b, args.effort)
    if suite.version_of(first["suite"]) != suite.version_of(second["suite"]):
        fail(f"{first['id']} and {second['id']} ran different suite versions, so they answered different questions")
    definition = definition_of(first)
    print(f"A = {first['model']} ({first['id']})\nB = {second['model']} ({second['id']})\n"
          "Paired over the questions both answered; the interval is a 95% bootstrap on A − B.\n")
    header = f"  {'':<22} {'A':>6} {'B':>6} {'A − B':>7}  {'95% interval':<17} n"
    print(header)
    by_task = {}
    for task in definition["index_tasks"]:
        scores_a = metrics.run_question_scores(paths.RUNS / first["id"], task)
        scores_b = metrics.run_question_scores(paths.RUNS / second["id"], task)
        by_task[task] = (scores_a, scores_b)
        entry = metrics.paired(scores_a, scores_b)
        if entry is None:
            print(f"  {definition['labels'][task]:<22} {'–':>6} {'–':>6}")
            continue
        interval = f"{entry['lo']:+.1f} to {entry['hi']:+.1f}"
        print(f"  {definition['labels'][task]:<22} {entry['a']:>6.1f} {entry['b']:>6.1f} {entry['diff']:>+7.1f}  "
              f"{interval:<17} {entry['n']:<4} {verdict(entry)}")
    index = metrics.paired_index({task: pair[0] for task, pair in by_task.items()},
                                 {task: pair[1] for task, pair in by_task.items()}, definition["groups"])
    if index:
        print(f"\n  {'Quality index':<22} {'':>6} {'':>6} {index['diff']:>+7.1f}  "
              f"{index['lo']:+.1f} to {index['hi']:+.1f}{'':<4} {verdict(index)}")


def cmd_doctor(args):
    profile = resolve_profile(args.model)
    try:
        level = resolve_effort(profile, args.effort)
    except ValueError as error:
        fail(str(error))
    host = hosts.for_profile(profile)
    if not host.reachable():
        fail(f"{'mbenchd' if profile.phone else 'llama-swap'} is not answering at {host.base_url()}")
    db = store.connect()
    reconcile(db)
    active = [run for run in store.list_runs(db) if run["status"] in ACTIVE]
    if active:
        fail(f"{active[0]['id']} is running; the checks would compete with it")
    others = [] if profile.phone else [entry["model"] for entry in swap.running() if entry["model"] != profile.id]
    if others:
        print(f"llama-swap will unload {', '.join(others)} to make room.")
    seconds = host.ensure_loaded()
    info = host.server_info()
    build = host.build(info)
    print(f"Loaded {profile.id} in {seconds} s" + (f", served by {stack.build_label(build)}" if stack.build_label(build) else "") + ".")
    capacity = swap.capacity(info)
    context = swap.positive(profile.context or capacity["context"], capacity["context"])
    moved = stack.changed(store.last_complete(db, profile.id), host.describe(), build)
    checks = asyncio.run(doctor.run(profile, level, context, capacity, moved))
    for check in checks:
        print(f"  {check['status'].upper():<4}  {check['check']:<12} {check['detail']}")
    if doctor.failures(checks):
        sys.exit(1)


def cmd_sources(_args):
    try:
        found = sources.check()
    except Exception as error:
        fail(f"couldn't check Hugging Face: {error!r}")
    for line in found["changed"]:
        print(f"changed  {line}")
    for line in found["newer"]:
        print(f"newer    {line}")
    if not found["changed"] and not found["newer"]:
        print("Every pinned file is current, and nothing newer is out.")


def cmd_board(args):
    path = board.build()
    print(path)
    if args.open:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)


def cmd_profile(args):
    profile = resolve_profile(args.model)
    for key, value in profile.to_dict().items():
        if key in ("sources", "cmd"):
            continue
        source = profile.sources.get(key)
        print(f"{key:<13} {value if value not in (None, {}, '') else '–'}" + (f"   ({source})" if source else ""))
    print(f"{'cmd':<13} {profile.cmd}")
    if profile.thinking == "none":
        print(f"\nNo thinking style is set, so --effort is not passed to this model. Add thinking = \"openai\" or \"qwen\""
              f" (and efforts = [...]) under [{profile.id}] in {paths.PROFILES}.")
    missing = lmx.missing_fields(profile)
    if missing:
        print(f"\n--submit needs {', '.join(missing)} under [{profile.id}] in {paths.PROFILES}")


def cmd_tick(_args):
    started = schedule.tick(store.connect())
    if started:
        print(f"started {started}")


def cmd_worker(args):
    from .worker import execute

    execute(args.run_id)


ROOT_EPILOG = """\
examples:
  mbench run qwen3-32b                          full suite at medium effort (2–5 h, in the background)
  mbench run qwen3-32b --effort max --submit
                                                the model at its maximum effort, recorded and submitted
  mbench run qwen3-32b --quick                  45–90 min, ranked as provisional
  mbench run qwen3-32b gpt-oss-120b --at 03:00 --until 08:00
                                                both models, one after the other, only at night
  mbench status                                 what is running and how far along it is
  mbench ls --effort max                        ranked table for one effort level
  mbench compare qwen3-32b gpt-oss-120b         which differences are real, task by task
  mbench doctor qwen3-32b                       check a model's server before spending a night on it
  mbench sources                                newer question sets, or pinned files that moved upstream
  mbench board --open                           the leaderboard page
  mbench export --out site                      the page plus board.json, ready to publish

A model is any llama-swap id (see `mbench profile <id>`). Only one run at a time; a run
swaps its model into the GPU and unloads whatever llama-swap had loaded.
Run `mbench run -h` for everything a run does and every option.
"""

RUN_DESCRIPTION = """\
Benchmark one llama-swap model and rank it on the leaderboard.

A run: waits for the GPU to be idle, loads the model through llama-swap, measures speed
(greedy: 1-request decode on 3 prompts, 1/2/4 requests at once, first-token wait from 1k
to 250k-token prompts, board power), then quality at the chosen effort (SuperGPQA, AIME
and HMMT 2026, agentic tool use, MRCR and Graphwalks long context, LiveCodeBench graded
in a sandbox), and rebuilds the leaderboard. Answers are saved as they arrive, so
`mbench resume` continues a stopped run. It stops itself and unloads the model if free
RAM drops under 4 GB.
"""

RUN_EPILOG = """\
suites:
  (default)  full: ~2 h for a fast MoE model, up to ~5 h for a verbose dense 27B at medium
  --quick    45–90 min, smaller samples, shown as provisional
  --smoke    a few items per task, checks the pipeline, never shown on the board

effort (--effort LEVEL, default medium):
  How hard the model reasons. max and min pick the top or bottom of the levels the model
  declares (efforts = [...] in models.toml, or Oh My Pi's thinking.efforts), so it is
  "xhigh" on a Qwen3.x template that declares it and "high" on gpt-oss. Named levels work;
  `mbench profile <id>` lists them. Each effort gets its own ranking on the board and in
  `mbench ls --effort`, so a max run sits next to the everyday medium one. Higher effort
  means more tokens per answer; a full max run of a verbose dense model can take 8 h+.

scheduling (--at TIME [--until TIME]):
  --at 03:00 (or --at "2026-09-12 03:00") starts the runs then instead of now; several
  models run one after another. --until 08:00 makes it a daily window: whatever is still
  running at 08:00 stops, llama-swap unloads the model, and the run continues from its
  last saved answer at 03:00 the next day. A user timer (mbench-tick.timer) checks every
  five minutes and switches itself off when nothing is scheduled; it keeps working after
  a reboot if lingering is on (loginctl enable-linger). Without --at, several models
  queue behind each other, and a run started while another is going waits its turn.
  Runs with --at give way to other GPU work: when a game or ComfyUI holds the GPU for a
  minute, the run stops, llama-swap unloads, and it tries again every ten minutes. Near
  the end of a window a run stops taking new questions that couldn't finish in time.

every run:
  starts with the checks `mbench doctor` runs (context per request, reasoning and
  tool-call parsing, a 16k-token prompt), sends as many requests at once as the server
  takes and as fit its shared context, and notifies the desktop, plus the `notify`
  command in ~/.config/mbench/config.toml or MBENCH_NOTIFY, when it finishes or fails.

reuse (--reuse [RUN]):
  Copies the answers of an earlier run of the same model, config, effort and suite size
  for every task whose questions and scoring haven't changed since, and measures only
  the rest. Without a run id it takes the newest run that qualifies.

localmaxxing (--submit [all|speed|evals], needs hf_id and quantization in
~/.config/mbench/models.toml):
  speed  lmx measures the two canonical prompts, then mbench submits them through the
         API with the prompt hash, output sample and timings, so they earn Verified
  evals  GSM8K and HellaSwag shards the site doesn't have yet for this model and
         quantization, answered at the run's effort
  all    both; what a bare --submit means

tasks (for --only/--skip): speed, supergpqa, math, tools, mrcr, graphwalks, lcb
  The quality index needs every quality task. It averages five groups equally:
  knowledge (supergpqa), math, code (lcb), long context (mrcr and graphwalks) and
  tool use.

examples:
  mbench run gpt-oss-120b                        full suite; Ctrl-C detaches, the run continues
  mbench run gpt-oss-120b --effort max --submit  maximum effort, full suite, everything submitted
  mbench run qwen3-32b --quick --effort max      a quick look at maximum effort
  mbench run gpt-oss-120b --reuse                keep what still applies from the last run
  mbench run qwen3-32b gpt-oss-120b --at 01:00 --until 07:30
                                                 both at night, paused by day until done
  mbench run qwen3-32b --only speed --submit speed
                                                 speed only, submitted as verified runs
  mbench run qwen3-32b --skip lcb --detach       everything except LiveCodeBench, return at once
  mbench run qwen3-32b --submit evals            full suite plus GSM8K/HellaSwag shards
"""


def parser():
    formatter = argparse.RawDescriptionHelpFormatter
    root = argparse.ArgumentParser(prog="mbench", description="Benchmark llama-swap models and rank them on one leaderboard.",
                                   epilog=ROOT_EPILOG, formatter_class=formatter)
    root.add_argument("--version", action="version", version=f"mbench {__version__}")
    commands = root.add_subparsers(dest="command", required=True, title="commands",
                                   metavar="{run,status,logs,cancel,resume,ls,compare,doctor,sources,export,board,profile}")

    run = commands.add_parser("run", help="benchmark a llama-swap model", description=RUN_DESCRIPTION,
                              epilog=RUN_EPILOG, formatter_class=formatter)
    run.add_argument("model", nargs="+", help="llama-swap model ids or aliases, run one after another")
    size = run.add_mutually_exclusive_group()
    size.add_argument("--quick", action="store_true", help="smaller samples (45–90 min), ranked as provisional")
    size.add_argument("--smoke", action="store_true", help="a few items per task (~10 min) to check the pipeline; never ranked")
    run.add_argument("--effort", default="medium", metavar="LEVEL",
                     help="reasoning effort: max, min, none, or a level the model declares (default medium); ranked per effort")
    run.add_argument("--only", metavar="TASKS", help="comma-separated tasks to run, e.g. speed or math,tools")
    run.add_argument("--skip", metavar="TASKS", help="comma-separated tasks to leave out, e.g. lcb")
    run.add_argument("--quality-from", metavar="RUN",
                     help="for a phone run: the desktop run of the same weights whose quality the board shows")
    run.add_argument("--reuse", nargs="?", const="latest", metavar="RUN",
                     help="carry over answers that still apply from an earlier run (default: the newest that qualifies)")
    run.add_argument("--submit", nargs="?", const="all", choices=lmx.SUBMIT_CHOICES, metavar="{all,speed,evals}",
                     help="also submit to localmaxxing: all (default), speed or evals")
    run.add_argument("--at", metavar="TIME", help="start at this time (03:00, or '2026-09-12 03:00') instead of now")
    run.add_argument("--keep-gpu", action="store_true",
                     help="don't step aside when another program wants the GPU (a game, ComfyUI)")
    run.add_argument("--until", metavar="TIME", help="with --at: stop at this time each day and continue at --at the next")
    run.add_argument("--note", help="free text stored with the run")
    run.add_argument("--detach", action="store_true", help="start and return immediately instead of following progress")
    run.add_argument("--foreground", action="store_true", help="run in this process instead of a systemd unit (debugging)")
    run.set_defaults(handler=cmd_run)

    commands.add_parser("status", help="show the run in progress and how far along it is").set_defaults(handler=cmd_status)
    logs = commands.add_parser("logs", help="print a run's log (latest run by default)")
    logs.add_argument("run", nargs="?", help="run id; defaults to the latest")
    logs.add_argument("-f", "--follow", action="store_true", help="keep printing until the run ends")
    logs.set_defaults(handler=cmd_logs)
    cancel = commands.add_parser("cancel", help="stop a run; `mbench resume` picks it up later")
    cancel.add_argument("run", nargs="?", help="run id; defaults to the latest")
    cancel.set_defaults(handler=cmd_cancel)
    resume = commands.add_parser("resume", help="continue a failed or cancelled run where it stopped")
    resume.add_argument("run", nargs="?", help="run id; defaults to the latest")
    resume.add_argument("--detach", action="store_true", help="start and return immediately")
    resume.add_argument("--foreground", action="store_true", help="run in this process instead of a systemd unit")
    resume.add_argument("--at", metavar="TIME", help="continue at this time instead of now")
    resume.add_argument("--keep-gpu", action="store_true", help="don't step aside for other GPU work")
    resume.add_argument("--until", metavar="TIME", help="with --at: stop at this time each day and continue the next")
    resume.set_defaults(handler=cmd_resume)
    ls = commands.add_parser("ls", help="ranked table of every model in the terminal")
    ls.add_argument("--effort", default="medium", metavar="LEVEL", help="which effort's ranking, e.g. medium or max (default medium)")
    ls.add_argument("--suite", metavar="VERSION", help=f"which suite version's ranking (default {suite.VERSION})")
    ls.add_argument("--markdown", action="store_true", help="print the table as markdown, for a README or an issue")
    ls.set_defaults(handler=cmd_ls)
    compare = commands.add_parser("compare", help="paired comparison of two models or runs, task by task",
                                  description="Compares two runs on the questions both answered, with a 95% bootstrap "
                                              "interval on each difference, so you can tell a real gap from noise.")
    compare.add_argument("a", help="model id (its headline run at --effort) or run id")
    compare.add_argument("b", help="model id or run id")
    compare.add_argument("--effort", default="medium", metavar="LEVEL", help="effort of the headline runs (default medium)")
    compare.set_defaults(handler=cmd_compare)
    doctor_parser = commands.add_parser("doctor", help="check a model's server: context, parsers, tool calls, a long prompt")
    doctor_parser.add_argument("model", help="llama-swap model id or alias")
    doctor_parser.add_argument("--effort", default="medium", metavar="LEVEL", help="effort to check at (default medium)")
    doctor_parser.set_defaults(handler=cmd_doctor)
    commands.add_parser("sources", help="newer question sets, or pinned files changed upstream").set_defaults(handler=cmd_sources)
    export_parser = commands.add_parser("export", help="write the leaderboard page and its data to a folder you can publish")
    export_parser.add_argument("--out", metavar="DIR", help="where to write index.html and board.json (default ~/.local/share/mbench/site)")
    export_parser.set_defaults(handler=cmd_export)
    board_parser = commands.add_parser("board", help="rebuild the leaderboard page and print its path")
    board_parser.add_argument("--open", action="store_true", help="also open it in the browser")
    board_parser.set_defaults(handler=cmd_board)
    profile = commands.add_parser("profile", help="what mbench knows about a model and where each fact came from")
    profile.add_argument("model", help="llama-swap model id or alias")
    profile.set_defaults(handler=cmd_profile)
    phone_parser = commands.add_parser("phone", help="the phone mbenchd runs on: forward its ports, push models, read its logs")
    phone_parser.add_argument("action", choices=("health", "launch", "forward", "push", "logs"), nargs="?", default="health")
    phone_parser.add_argument("files", nargs="*", help="for push: the .gguf files to copy into the app")
    phone_parser.add_argument("--into", help="for logs: where to write them (default: here)")
    phone_parser.set_defaults(handler=cmd_phone)
    commands.add_parser("tick").set_defaults(handler=cmd_tick)
    worker = commands.add_parser("worker")
    worker.add_argument("run_id")
    worker.set_defaults(handler=cmd_worker)
    return root


def main():
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
