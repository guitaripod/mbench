import statistics
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

SERVER_PROCESS_NAMES = ("sglang", "llama-server", "llama-swap", "vllm")
BUSY_SM_PERCENT = 25
HEAVY_MIB = 4096


class Sampler:
    """Samples board power, VRAM and utilisation on an interval so each measurement can report its own window."""

    def __init__(self, interval_ms=250):
        self.samples = []
        self.process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits", "-lms", str(interval_ms)],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.process.stdout or []:
            try:
                power, memory, utilization, temperature = (float(part) for part in line.split(","))
            except ValueError:
                continue
            self.samples.append((time.time(), power, memory, utilization, temperature))

    def window(self, start, end):
        inside = [sample for sample in self.samples if start <= sample[0] <= end]
        if not inside:
            return {}
        busy = [sample for sample in inside if sample[3] >= 30] or inside
        return {"power_w": round(statistics.mean(sample[1] for sample in busy), 1),
                "vram_mib": max(sample[2] for sample in inside),
                "temp_c": round(max(sample[4] for sample in inside), 1)}

    def timeline(self):
        """A card reports power, not a thermal state; the phone's timeline has no counterpart here."""
        return []

    def energy_wh(self, start, end):
        """Board energy between two moments from the power samples, idle draw included; None without enough samples."""
        points = [(sample[0], sample[1]) for sample in self.samples if start <= sample[0] <= end]
        if len(points) < 2:
            return None
        joules = sum((later[0] - earlier[0]) * (earlier[1] + later[1]) / 2 for earlier, later in zip(points, points[1:]))
        return joules / 3600

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
    """The executable's name, from a Linux path or a Windows one (games under Proton report S:\\...\\Game.exe)."""
    first = (process_name or "").split(" --")[0].strip()
    return first.replace("\\", "/").rsplit("/", 1)[-1] or "?"


def number(text):
    try:
        return float(text)
    except ValueError:
        return 0.0


def pmon_rows(text):
    """nvidia-smi pmon's table as dicts keyed by its own header, so older drivers with fewer columns still parse."""
    columns, rows = None, []
    for line in text.splitlines():
        if line.startswith("#"):
            columns = columns or line.lstrip("#").split()
            continue
        parts = line.split()
        if columns and len(parts) >= len(columns):
            head = len(columns) - 1
            rows.append({**dict(zip(columns[:head], parts[:head])), columns[-1]: " ".join(parts[head:])})
    return rows


def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def parent_of(pid):
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def is_server(pid):
    """A model server or anything llama-swap started: the benchmark's own GPU work, never contention."""
    for _ in range(64):
        if not pid or pid <= 1:
            return False
        if any(name in cmdline(pid) for name in SERVER_PROCESS_NAMES):
            return True
        pid = parent_of(pid)
    return False


def processes(samples=3):
    """Every GPU process with its SM use averaged over one-second samples and the VRAM it holds."""
    completed = subprocess.run(["nvidia-smi", "pmon", "-c", str(samples), "-s", "um"],
                               capture_output=True, text=True, timeout=30 + 2 * samples)
    sm, mib, names = defaultdict(list), defaultdict(float), {}
    for row in pmon_rows(completed.stdout):
        if not row.get("pid", "").isdigit():
            continue
        pid = int(row["pid"])
        sm[pid].append(number(row.get("sm", "-")))
        mib[pid] = max(mib[pid], number(row.get("fb", "-")))
        names[pid] = row.get("command", "")
    if not names:
        for app in compute_apps():
            sm[app["pid"]], mib[app["pid"]], names[app["pid"]] = [0.0], float(app["mib"]), app["name"]
    return [{"pid": pid, "name": short_name(cmdline(pid) or names[pid]), "sm": statistics.mean(sm[pid]),
             "mib": int(mib[pid]), "server": is_server(pid)} for pid in names]


def contention(samples=3):
    """Other GPU work big enough to matter: a process outside the model servers using a quarter of the GPU or holding
    4 GB, like a game or ComfyUI. A browser tab or the compositor stays under both."""
    return [process for process in processes(samples) if not process["server"]
            and (process["sm"] >= BUSY_SM_PERCENT or process["mib"] >= HEAVY_MIB)]


def describe_contention(found):
    return ", ".join(f"{process['name']} ({process['sm']:.0f}% GPU, {process['mib'] / 1024:.1f} GB)" for process in found)


def health():
    """The card as the driver sees it right now; None when nvidia-smi cannot reach it at all, which is what a GPU that
    has fallen off the bus looks like from here."""
    completed = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
    parts = [part.strip() for part in completed.stdout.split(",")]
    if completed.returncode != 0 or len(parts) != 2:
        return None
    try:
        return {"temp_c": float(parts[0]), "power_w": float(parts[1])}
    except ValueError:
        return None


def last_fault(minutes=10):
    """The newest Xid the kernel logged, which names what the driver hit when the card stopped answering."""
    completed = subprocess.run(["journalctl", "-k", "--since", f"-{minutes}min", "-o", "cat", "--no-pager"],
                               capture_output=True, text=True)
    faults = [line for line in completed.stdout.splitlines() if "Xid" in line]
    return faults[-1].split("NVRM: ")[-1] if faults else None


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
