import json
import shutil
import subprocess

from . import datasets, paths

IMAGE = "mbench-lcb-grader"
LCB_REPO = "https://github.com/LiveCodeBench/LiveCodeBench.git"
LCB_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
GRADER_DIR = paths.PACKAGE / "grader"


def ensure_image():
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0:
        return
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, str(GRADER_DIR)], check=True, capture_output=True)


def head(repo):
    completed = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    return completed.stdout.strip()


def ensure_runner():
    """LiveCodeBench's own test harness at a pinned commit; the code runs only inside the network-less container."""
    target = paths.CACHE / "LiveCodeBench"
    if (target / "lcb_runner").exists() and head(target) == LCB_COMMIT:
        return target
    shutil.rmtree(target, ignore_errors=True)
    local = paths.LEGACY_BENCH / "lcb"
    if (local / "lcb_runner").exists() and head(local) == LCB_COMMIT:
        shutil.copytree(local, target)
        return target
    subprocess.run(["git", "clone", "--quiet", LCB_REPO, str(target)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--quiet", LCB_COMMIT], check=True, capture_output=True)
    return target


def grade(run_dir):
    results = run_dir / "lcb.jsonl"
    if not results.exists() or not results.read_text().strip():
        return {}
    ensure_image()
    runner = ensure_runner()
    dataset = datasets.fetch("lcb")
    command = [
        "docker", "run", "--rm", "--network", "none", "--cpus", "8", "--memory", "8g", "--pids-limit", "4096",
        "-v", f"{runner}:/lcb:ro", "-v", f"{dataset.parent}:/data:ro",
        "-v", f"{GRADER_DIR / 'grade_lcb.py'}:/grade_lcb.py:ro", "-v", f"{run_dir}:/work",
        IMAGE, "python", "/grade_lcb.py", "/work/lcb.jsonl", "/work/lcb.graded.jsonl", f"/data/{dataset.name}",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=7200)
    with (run_dir / "lcb.graded.jsonl").open() as handle:
        return {row["id"]: row["passed"] for row in map(json.loads, handle)}
