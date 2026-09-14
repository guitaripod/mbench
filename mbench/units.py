import re
import subprocess
import sys
import time

from . import paths, store

ACTIVE = ("queued", "running")
GPU_CLASS = "gpu"


def device_class(run):
    return ((run or {}).get("hardware") or {}).get("class") or GPU_CLASS


def busy(runs, klass=GPU_CLASS):
    """Runs competing for the same device. A phone and the card are separate machines, so a run on one never has
    to wait for a run on the other."""
    return [run for run in runs if run["status"] in ACTIVE and device_class(run) == klass]


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


def stop(run_id):
    subprocess.run(["systemctl", "--user", "stop", unit_name(run_id)], capture_output=True)
