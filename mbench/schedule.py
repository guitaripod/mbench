import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from . import gpu, notify, paths, store, swap, units

TIMER = "mbench-tick"
UNIT_DIR = Path(os.environ.get("XDG_CONFIG_HOME", paths.HOME / ".config")) / "systemd" / "user"
PASSED_ENV = ("XDG_CACHE_HOME", "XDG_CONFIG_HOME")


def parse_clock(text):
    """HH:MM on a 24-hour clock, returned zero-padded; a ValueError names what was wrong."""
    try:
        moment = datetime.strptime(text.strip(), "%H:%M")
    except ValueError:
        raise ValueError(f"'{text}' is not a time like 03:00") from None
    return moment.strftime("%H:%M")


def parse_at(text):
    """A clock time (the next time it comes round) or a full date and time; returns (first start or None, clock)."""
    for layout in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            moment = datetime.strptime(text.strip(), layout)
        except ValueError:
            continue
        return moment, moment.strftime("%H:%M")
    return None, parse_clock(text)


def minute_of(clock):
    hours, minutes = map(int, clock.split(":"))
    return hours * 60 + minutes


def in_window(moment, window):
    """Whether a moment falls in a daily window; a window without an end never closes, and one like 23:00–07:00 spans midnight."""
    if not window or not window.get("end"):
        return True
    start, end, now = minute_of(window["start"]), minute_of(window["end"]), moment.hour * 60 + moment.minute
    return start <= now < end if start < end else now >= start or now < end


def next_start(moment, clock):
    hours, minutes = map(int, clock.split(":"))
    candidate = moment.replace(hour=hours, minute=minutes, second=0, microsecond=0)
    return candidate if candidate >= moment else candidate + timedelta(days=1)


def window_deadline(window, moment):
    """When the window that is open now closes; None for a window without an end, or when it isn't open."""
    if not window or not window.get("end") or not in_window(moment, window):
        return None
    return next_start(moment, window["end"]).timestamp()


def first_start(moment, window, at=None):
    """When a newly scheduled run may begin: the given date, now if already inside its window, else the next window start."""
    if at is not None:
        return max(at, moment)
    if window.get("end") and in_window(moment, window):
        return moment
    return next_start(moment, window["start"])


def describe(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%a %d %b %H:%M")


def unit_files():
    environment = {key: value for key, value in os.environ.items() if key.startswith("MBENCH_") or key in PASSED_ENV}
    service = (
        "[Unit]\nDescription=Start due mbench runs and pause runs outside their window\n\n[Service]\nType=oneshot\n"
        + "".join(f'Environment="{key}={value}"\n' for key, value in sorted(environment.items()))
        + f"WorkingDirectory={paths.DATA}\nExecStart={sys.executable} -m mbench.cli tick\n"
    )
    timer = ("[Unit]\nDescription=Check the mbench schedule every five minutes\n\n[Timer]\nOnCalendar=*:0/5\n"
             "AccuracySec=1s\n\n[Install]\nWantedBy=timers.target\n")
    return {f"{TIMER}.service": service, f"{TIMER}.timer": timer}


def install_timer():
    """A persistent user timer, so a schedule survives reboots and logouts (with lingering on)."""
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    changed = False
    for name, text in unit_files().items():
        path = UNIT_DIR / name
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
            changed = True
    if changed:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", f"{TIMER}.timer"], check=True, capture_output=True)


def remove_timer():
    subprocess.run(["systemctl", "--user", "disable", "--now", f"{TIMER}.timer"], capture_output=True)


def note(run_id, message):
    log = paths.RUNS / run_id / "worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def gives_way(flags):
    """Every run steps aside for a game or another GPU job unless it was started with --keep-gpu."""
    return flags.get("yield", True)


def park(db, run_id, not_before, reason, counter):
    """Puts a run back on the schedule to continue from its saved answers; counter is "paused" for a closing window and
    "yielded" for giving the GPU away. Returns how many times that has happened to the run."""
    flags = (store.get_run(db, run_id) or {}).get("flags") or {}
    count = flags.get(counter, 0) + 1
    store.update_run(db, run_id, status="scheduled",
                     flags={**flags, "not_before": not_before, counter: count, "parked": reason})
    note(run_id, f"paused: {reason}; continues {describe(not_before)} at the earliest")
    return count


def pause(db, run, moment):
    """Stops a run whose window closed and frees the GPU; its saved answers carry it on when the window next opens."""
    window = (run.get("flags") or {})["window"]
    units.stop(run["id"])
    swap.unload()
    resumes = next_start(moment, window["start"]).timestamp()
    if park(db, run["id"], resumes, f"the window closed at {window['end']}", "paused") == 1:
        notify.send("mbench paused", f"{run['model']}: the window closed at {window['end']}; continues {describe(resumes)}",
                    run=run["id"], status="paused")


def due(run, moment):
    return ((run.get("flags") or {}).get("not_before") or 0) <= moment.timestamp()


def tick(db, moment=None):
    """Pauses runs whose window has closed, then starts the first due scheduled run if nothing is running; returns its id."""
    moment = moment or datetime.now()
    units.reconcile(db)
    for run in store.list_runs(db):
        if run["status"] in units.ACTIVE and not in_window(moment, (run.get("flags") or {}).get("window")):
            pause(db, run, moment)
    runs = store.list_runs(db)
    started = None
    if not any(run["status"] in units.ACTIVE for run in runs):
        waiting = sorted((run for run in runs if run["status"] == "scheduled" and due(run, moment)),
                         key=lambda run: ((run.get("flags") or {}).get("not_before") or 0, run.get("created") or 0))
        contended = None
        for run in waiting:
            flags = run.get("flags") or {}
            if not in_window(moment, flags.get("window")):
                later = next_start(moment, flags["window"]["start"]).timestamp()
                store.update_run(db, run["id"], flags={**flags, "not_before": later})
                continue
            if gives_way(flags):
                contended = gpu.contention() if contended is None else contended
                if contended:
                    continue
            store.update_run(db, run["id"], status="queued", error=None)
            units.spawn(run["id"], False)
            started = run["id"]
            break
    if not any(run["status"] == "scheduled" for run in store.list_runs(db)):
        remove_timer()
    return started
