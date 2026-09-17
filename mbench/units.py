import re
import subprocess
import sys
import time

from . import paths, store

ACTIVE = ("queued", "running")
GPU_CLASS = "gpu"
MAX_SHARED_RUNS = 3
RECONCILE_GRACE_S = 300


def class_of(device):
    """The hardware class a run of this kind records. A remote server is named by the machine it runs on, which is
    what the board groups and compares within."""
    return {"remote": "mac"}.get(device, device)


def device_class(run):
    return ((run or {}).get("hardware") or {}).get("class") or GPU_CLASS


def measures_speed(run):
    return "speed" in ((run.get("flags") or {}).get("tasks") or ["speed"])


def busy(runs, klass=GPU_CLASS, speed=True):
    """Runs competing for the same device. A phone and the card are separate machines, so a run on one never waits
    for a run on the other — and two runs that score no speed never have to wait either, because nothing they
    measure changes when they share the hardware. Only a speed measurement needs the device to itself."""
    active = [run for run in runs if run["status"] in ACTIVE and device_class(run) == klass]
    if speed:
        return active
    sharing = [run for run in active if not measures_speed(run)]
    blocking = [run for run in active if measures_speed(run)]
    return blocking or (sharing if len(sharing) >= MAX_SHARED_RUNS else [])


def unit_name(run_id):
    return "mbench-" + re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)


def unit_active(run_id):
    return subprocess.run(["systemctl", "--user", "is-active", "-q", unit_name(run_id)]).returncode == 0


def reconcile(db):
    """A worker that died without writing its verdict (reboot, kill) leaves a running row; mark it failed so it can be
    resumed. The age of the row is not evidence of that — a run queued an hour ago and spawned a second ago is young
    in every way that matters — so a row is only given up on once its unit has been gone for a while."""
    for run in store.list_runs(db):
        if run["status"] not in ACTIVE or unit_active(run["id"]):
            continue
        idle = time.time() - (run.get("started") or run.get("created") or 0)
        if idle > RECONCILE_GRACE_S:
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


def stop(run_id):
    subprocess.run(["systemctl", "--user", "stop", unit_name(run_id)], capture_output=True)
