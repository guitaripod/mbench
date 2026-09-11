import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import NoReturn

from . import __version__, board, gpu, lmx, paths, profiles, store, suite, swap

ACTIVE = ("queued", "running")
DURATIONS = {"full": "1–4 hours", "quick": "30–60 minutes", "smoke": "about 5 minutes"}


def fail(message) -> NoReturn:
    print(f"mbench: {message}", file=sys.stderr)
    sys.exit(1)


def harness():
    root = paths.PACKAGE.parent
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "mbench"],
                           capture_output=True, text=True).stdout.strip()
    return f"{__version__}+{commit or 'nogit'}{'-dirty' if dirty else ''}"


def unit_name(run_id):
    return "mbench-" + re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)


def unit_active(run_id):
    return subprocess.run(["systemctl", "--user", "is-active", "-q", unit_name(run_id)]).returncode == 0


def reconcile(db):
    """A worker that died without writing its verdict (reboot, kill) leaves a running row; mark it failed so it can be resumed."""
    for run in store.list_runs(db):
        if run["status"] in ACTIVE and not unit_active(run["id"]) and time.time() - (run["created"] or 0) > 60:
            store.update_run(db, run["id"], status="failed", error="the worker stopped without finishing")


def spawn(run_id, foreground):
    """Runs the worker as a transient systemd unit so closing the terminal or ending an agent session can't kill it."""
    if foreground:
        from .worker import execute

        execute(run_id)
        return
    subprocess.run(["systemctl", "--user", "reset-failed", unit_name(run_id)], capture_output=True)
    subprocess.run(
        ["systemd-run", "--user", "--collect", f"--unit={unit_name(run_id)}", "-p", "MemoryMax=8G",
         f"--working-directory={paths.DATA}", sys.executable, "-m", "mbench.cli", "worker", run_id],
        check=True, capture_output=True,
    )


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
            text = f"{stamp}  {phase:<8} [{'#' * filled}{'.' * (24 - filled)}] {event['done']}/{event['total']}  eta {duration(eta)}"
            if self.tty:
                print("\r" + text.ljust(78), end="", flush=True)
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
        print(f"{stamp}  {phase:<8} {details}".rstrip())


def summary(db, run_id):
    metrics = store.metrics_of(db, run_id)

    def show(key, label, unit=""):
        entry = metrics.get(key)
        if entry and entry.get("value") is not None:
            interval = f"  ({entry['lo']:.1f}–{entry['hi']:.1f})" if entry.get("lo") is not None and unit == "%" else ""
            print(f"  {label:<26} {entry['value']:>8.1f} {unit}{interval}")

    show("index.quality", "Quality index", "%")
    for task in suite.INDEX_TASKS:
        show(f"{task}.score", suite.TASK_LABELS[task], "%")
    show("speed.decode", "Decode, one request", "tok/s")
    show("speed.conc.4", "Throughput, 4 at once", "tok/s")
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


def cmd_run(args):
    try:
        profile = profiles.resolve(args.model)
    except KeyError as error:
        fail(str(error.args[0]))
    if not swap.reachable():
        fail(f"llama-swap is not answering at {paths.SWAP_URL}")
    tasks = ["speed", *suite.QUALITY_TASKS]
    chosen = set(args.only.split(",")) if args.only else set(tasks)
    skipped = set(args.skip.split(",")) if args.skip else set()
    unknown = (chosen | skipped) - set(tasks)
    if unknown:
        fail(f"unknown task {', '.join(sorted(unknown))}; tasks are {', '.join(tasks)}")
    tasks = [task for task in tasks if task in chosen and task not in skipped]
    if args.submit and lmx.missing_fields(profile):
        fail(f"--submit needs {', '.join(lmx.missing_fields(profile))} for {profile.id} in {paths.PROFILES}")
    db = store.connect()
    reconcile(db)
    active = [run for run in store.list_runs(db) if run["status"] in ACTIVE]
    if active:
        fail(f"{active[0]['id']} is still running; follow it with `mbench status` or stop it with `mbench cancel`")
    suite_name = "smoke" if args.smoke else "quick" if args.quick else "full"
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{profile.id}"
    (paths.RUNS / run_id).mkdir(parents=True, exist_ok=True)
    store.insert_run(db, {
        "id": run_id, "model": profile.id, "name": profile.name, "suite": suite.label(suite_name),
        "effort": args.effort, "status": "queued", "harness": harness(), "fingerprint": profile.fingerprint,
        "profile": profile.to_dict(), "hardware": gpu.describe(), "flags": {"tasks": tasks, "submit": args.submit},
        "note": args.note,
    })
    others = [entry["model"] for entry in swap.running() if entry["model"] != profile.id]
    print(f"{run_id}: {suite.label(suite_name)} on {profile.name}, usually {DURATIONS[suite_name]}.")
    if others:
        print(f"llama-swap will unload {', '.join(others)} to make room.")
    spawn(run_id, args.foreground)
    if not args.foreground and not args.detach:
        follow(run_id)


def latest_run(db, run_id=None):
    if run_id:
        return store.get_run(db, run_id) or fail(f"no run {run_id}")
    runs = store.list_runs(db)
    return runs[0] if runs else fail("no runs yet; start one with `mbench run <llama-swap model>`")


def cmd_status(_args):
    db = store.connect()
    reconcile(db)
    active = [run for run in store.list_runs(db) if run["status"] in ACTIVE]
    if not active:
        print("No run in progress.")
        runs = store.list_runs(db)
        if runs:
            last = runs[0]
            print(f"Last: {last['id']} {last['status']}" + (f" ({last['error']})" if last.get("error") else ""))
        return
    for run in active:
        events = paths.RUNS / run["id"] / "events.jsonl"
        last = json.loads(events.read_text().splitlines()[-1]) if events.exists() and events.read_text().strip() else {}
        detail = f"{last.get('phase', 'queued')} {last.get('done', '')}/{last.get('total', '')}".rstrip("/ ")
        print(f"{run['id']}  {run['suite']}  {detail}  running for {duration(time.time() - (run['started'] or run['created']))}")


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


def cmd_resume(args):
    db = store.connect()
    reconcile(db)
    run = latest_run(db, args.run)
    if run["status"] in ACTIVE:
        fail(f"{run['id']} is already running")
    if run["status"] == "complete":
        fail(f"{run['id']} already finished")
    store.update_run(db, run["id"], status="queued", error=None)
    spawn(run["id"], args.foreground)
    if not args.foreground and not args.detach:
        follow(run["id"])


def cell(entry, digits=1):
    if not entry or entry.get("value") is None:
        return "–"
    return f"{entry['value']:.{digits}f}"


def cmd_ls(_args):
    data = board.collect(store.connect())
    if not data["models"]:
        print("No finished runs yet. Start one with `mbench run <llama-swap model>`.")
        return
    header = ["Model", "Quality", *[suite.TASK_LABELS[task].split()[0] for task in suite.INDEX_TASKS],
              "tok/s", "4× tok/s", "TTFT 32k", "Run"]
    rows = []
    ordered = sorted(data["models"], key=lambda model: -((model["index"] or {}).get("value") or -1))
    for model in ordered:
        speed = model["speed"]
        kind = model["run"]["kind"]
        rows.append([
            model["id"], cell(model["index"]), *[cell(model["tasks"].get(task)) for task in suite.INDEX_TASKS],
            cell(speed.get("decode"), 0), cell(speed.get("conc.4"), 0), cell(speed.get("ttft.32000")),
            datetime.fromtimestamp(model["run"]["finished"]).strftime("%d %b") + ("" if kind == "full" else f" {kind}"),
        ])
    widths = [max(len(str(row[index])) for row in [header, *rows]) for index in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(str(value).ljust(width) if index == 0 else str(value).rjust(width)
                        for index, (value, width) in enumerate(zip(row, widths))))


def cmd_board(args):
    path = board.build()
    print(path)
    if args.open:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)


def cmd_profile(args):
    try:
        profile = profiles.resolve(args.model)
    except KeyError as error:
        fail(str(error.args[0]))
    for key, value in profile.to_dict().items():
        if key in ("sources", "cmd"):
            continue
        source = profile.sources.get(key)
        print(f"{key:<13} {value if value not in (None, {}, '') else '–'}" + (f"   ({source})" if source else ""))
    print(f"{'cmd':<13} {profile.cmd}")
    missing = lmx.missing_fields(profile)
    if missing:
        print(f"\n--submit needs {', '.join(missing)} under [{profile.id}] in {paths.PROFILES}")


def cmd_import_legacy(_args):
    from .legacy import import_all

    imported = import_all(store.connect())
    print("\n".join(imported) if imported else "Nothing new to import.")
    print(board.build())


def cmd_worker(args):
    from .worker import execute

    execute(args.run_id)


ROOT_EPILOG = """\
examples:
  mbench run qwen38-nvfp4                 full suite at medium effort (1–4 h, runs in the background)
  mbench run qwen38-nvfp4 --quick         30–60 min, ranked as provisional
  mbench run qwen38-nvfp4 --smoke         ~5 min pipeline check, never ranked
  mbench status                           what is running and how far along it is
  mbench ls                               ranked table in the terminal
  mbench board --open                     the leaderboard page

A model is any llama-swap id (see `mbench profile <id>`). Only one run at a time; a run
swaps its model into the GPU and unloads whatever llama-swap had loaded.
Run `mbench run -h` for everything a run does and its options.
"""

RUN_DESCRIPTION = """\
Benchmark one llama-swap model and rank it on the leaderboard.

A run: waits for the GPU to be idle, loads the model through llama-swap, measures speed
(greedy: 1-request decode on 3 prompts, 1/2/4 requests at once, first-token wait from 1k
to 250k-token prompts, board power), then quality at the chosen effort (needle retrieval,
MMLU-Pro, AIME 2025 ×4, tool calls, LiveCodeBench graded in a sandbox), and rebuilds the
leaderboard. Answers are saved as they arrive, so `mbench resume` continues a stopped run.
It stops itself and unloads the model if free RAM drops under 4 GB.
"""

RUN_EPILOG = """\
suites:
  (default)  full: ~1.5 h for a gpt-oss-sized model, up to ~4 h for a verbose 27B
  --quick    30–60 min, smaller samples, shown as provisional
  --smoke    a few items per task, checks the pipeline, never shown on the board

effort:
  The leaderboard ranks medium-effort runs only, so every model is compared the same way.
  low/high runs are kept in the model's history. Qwen templates get "xhigh" for high.

tasks (for --only/--skip): speed, niah, mmlupro, aime, tools, lcb
  The quality index needs niah, mmlupro, aime, tools and lcb all present.

examples:
  mbench run sglang-gptoss120b                    full suite; Ctrl-C detaches, the run continues
  mbench run qwen38-nvfp4 --quick --detach        start and return immediately
  mbench run qwen38-nvfp4 --effort high           same suite at high effort (history only)
  mbench run sglang-27b --only speed              just the speed tests
  mbench run sglang-27b --skip lcb                everything except LiveCodeBench
  mbench run sglang-gptoss120b --submit           also submit to localmaxxing (needs hf_id and
                                                  quantization in ~/.config/mbench/models.toml)
"""


def parser():
    formatter = argparse.RawDescriptionHelpFormatter
    root = argparse.ArgumentParser(prog="mbench", description="Benchmark llama-swap models and rank them on one leaderboard.",
                                   epilog=ROOT_EPILOG, formatter_class=formatter)
    root.add_argument("--version", action="version", version=f"mbench {__version__}")
    commands = root.add_subparsers(dest="command", required=True, title="commands",
                                   metavar="{run,status,logs,cancel,resume,ls,board,profile,import-legacy}")

    run = commands.add_parser("run", help="benchmark a llama-swap model", description=RUN_DESCRIPTION,
                              epilog=RUN_EPILOG, formatter_class=formatter)
    run.add_argument("model", help="llama-swap model id or alias, e.g. qwen38-nvfp4")
    size = run.add_mutually_exclusive_group()
    size.add_argument("--quick", action="store_true", help="smaller samples (30–60 min), ranked as provisional")
    size.add_argument("--smoke", action="store_true", help="a few items per task (~5 min) to check the pipeline; never ranked")
    run.add_argument("--effort", default="medium", choices=("low", "medium", "high"),
                     help="reasoning effort (default medium; only medium is ranked)")
    run.add_argument("--only", metavar="TASKS", help="comma-separated tasks to run, e.g. speed or niah,mmlupro")
    run.add_argument("--skip", metavar="TASKS", help="comma-separated tasks to leave out, e.g. lcb")
    run.add_argument("--submit", action="store_true", help="also submit localmaxxing speed runs and GSM8K/HellaSwag shards")
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
    resume.set_defaults(handler=cmd_resume)
    commands.add_parser("ls", help="ranked table of every model in the terminal").set_defaults(handler=cmd_ls)
    board_parser = commands.add_parser("board", help="rebuild the leaderboard page and print its path")
    board_parser.add_argument("--open", action="store_true", help="also open it in the browser")
    board_parser.set_defaults(handler=cmd_board)
    profile = commands.add_parser("profile", help="what mbench knows about a model and where each fact came from")
    profile.add_argument("model", help="llama-swap model id or alias")
    profile.set_defaults(handler=cmd_profile)
    commands.add_parser("import-legacy", help="import the 11 Sep 2026 gpt-oss vs Qwen results").set_defaults(handler=cmd_import_legacy)
    worker = commands.add_parser("worker")
    worker.add_argument("run_id")
    worker.set_defaults(handler=cmd_worker)
    return root


def main():
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
