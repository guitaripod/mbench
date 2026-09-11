import statistics
import subprocess
import threading
import time
from pathlib import Path

SERVER_PROCESS_NAMES = ("sglang", "llama-server", "vllm")


class Sampler:
    """Samples board power, VRAM and utilisation every 250 ms so each measurement can report its own window."""

    def __init__(self):
        self.samples = []
        self.process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw,memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-lms", "250"],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.process.stdout or []:
            try:
                power, memory, utilization = (float(part) for part in line.split(","))
            except ValueError:
                continue
            self.samples.append((time.time(), power, memory, utilization))

    def window(self, start, end):
        inside = [sample for sample in self.samples if start <= sample[0] <= end]
        if not inside:
            return {}
        busy = [sample for sample in inside if sample[3] >= 30] or inside
        return {"power_w": round(statistics.mean(sample[1] for sample in busy), 1),
                "vram_mib": max(sample[2] for sample in inside)}

    def close(self):
        self.process.kill()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def compute_apps():
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    apps = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[2].isdigit():
            apps.append({"pid": int(parts[0]), "name": parts[1], "mib": int(parts[2])})
    return apps


def short_name(process_name):
    return Path(process_name.split()[0]).name if process_name else "?"


def heavy_apps(min_mib=4096):
    """Non-server GPU processes holding a lot of VRAM (ComfyUI, games); browsers and the compositor stay well under 4 GB."""
    return [{**app, "name": short_name(app["name"])} for app in compute_apps()
            if app["mib"] >= min_mib and not any(name in app["name"] for name in SERVER_PROCESS_NAMES)]


def utilization(seconds=5.0):
    """Average GPU utilisation over a few seconds; idle desktops sit near zero even with a browser open."""
    samples = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True)
        try:
            samples.append(float(completed.stdout.splitlines()[0]))
        except (ValueError, IndexError):
            pass
        time.sleep(0.5)
    return statistics.mean(samples) if samples else 0.0


def describe():
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    name, driver, memory = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
    return {"gpu": name, "driver": driver, "vram_mib": int(memory)}


def mem_available_gb():
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return 0.0
