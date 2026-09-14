import json
import os
import shutil
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

BUNDLE_SUFFIX = ".com.midgar.mbenchd"
CONTROL_PORT = 8081
SERVER_PORT = 8080
CONTROL_URL = "http://127.0.0.1:18081"
SERVER_URL = "http://127.0.0.1:18080"
LOAD_TIMEOUT_S = 1800
HEALTH_TIMEOUT_S = 10

MODELS = {
    "iPhone18,1": {"device": "iPhone 17 Pro", "soc": "A19 Pro", "ram_gb": 12, "bandwidth_gbs": 76.8},
    "iPhone18,2": {"device": "iPhone 17 Pro Max", "soc": "A19 Pro", "ram_gb": 12, "bandwidth_gbs": 76.8},
    "iPhone18,3": {"device": "iPhone 17", "soc": "A19", "ram_gb": 8, "bandwidth_gbs": 68.3},
    "iPhone18,4": {"device": "iPhone Air", "soc": "A19 Pro", "ram_gb": 12, "bandwidth_gbs": 68.3},
    "iPhone17,1": {"device": "iPhone 16 Pro", "soc": "A18 Pro", "ram_gb": 8, "bandwidth_gbs": 60.0},
    "iPhone17,2": {"device": "iPhone 16 Pro Max", "soc": "A18 Pro", "ram_gb": 8, "bandwidth_gbs": 60.0},
    "iPhone17,3": {"device": "iPhone 16", "soc": "A18", "ram_gb": 8, "bandwidth_gbs": 60.0},
    "iPhone16,1": {"device": "iPhone 15 Pro", "soc": "A17 Pro", "ram_gb": 8, "bandwidth_gbs": 51.2},
    "iPhone16,2": {"device": "iPhone 15 Pro Max", "soc": "A17 Pro", "ram_gb": 8, "bandwidth_gbs": 51.2},
}

THERMAL_ORDER = ("nominal", "fair", "serious", "critical")


class Unreachable(RuntimeError):
    """The app on the phone stopped answering. Its answers are missing, not wrong, so a run stops instead of
    scoring the silence."""


class Device:
    """The mbenchd app on the phone: it loads a model into llama-server, and reports thermal state, memory and
    battery while the benchmark runs."""

    def __init__(self, control_url=CONTROL_URL, server_url=SERVER_URL):
        self.control_url = control_url.rstrip("/")
        self.server_url = server_url.rstrip("/")

    def get(self, path, timeout=HEALTH_TIMEOUT_S):
        with urllib.request.urlopen(self.control_url + path, timeout=timeout) as response:
            return json.loads(response.read())

    def post(self, path, payload, timeout=HEALTH_TIMEOUT_S):
        request = urllib.request.Request(self.control_url + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    def health(self):
        try:
            return self.get("/mb/health")
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError):
            return None

    def reachable(self):
        return self.health() is not None

    def models(self):
        return (self.get("/mb/models") or {}).get("data") or []

    def load(self, request):
        """Loads the model and waits for llama-server to answer; the phone needs minutes for a first load, since
        Metal compiles every kernel from source on the device."""
        return self.post("/mb/load", request, timeout=LOAD_TIMEOUT_S)

    def unload(self):
        try:
            self.post("/mb/unload", {}, timeout=120)
        except (OSError, urllib.error.HTTPError):
            pass

    def describe(self):
        """The hardware blob a run records: what phone answered, and the bandwidth its decode speed is bounded by."""
        health = self.health()
        if health is None:
            raise Unreachable(f"no mbenchd at {self.control_url}")
        hardware = (health.get("device") or {}).get("hardware") or "unknown"
        known = MODELS.get(hardware, {})
        return {
            "class": "phone",
            "hardware": hardware,
            "device": known.get("device") or hardware,
            "soc": known.get("soc"),
            "ram_gb": known.get("ram_gb"),
            "bandwidth_gbs": known.get("bandwidth_gbs"),
            "os": (health.get("device") or {}).get("system"),
            "app": (health.get("app") or {}).get("version"),
        }

    def build(self):
        """The stack a phone run's numbers come from: the llama.cpp the app embeds, not the desktop's."""
        health = self.health() or {}
        app = health.get("app") or {}
        return {"engine": "llama.cpp", "version": f"b{app.get('llama_build')}",
                "commit": app.get("llama_commit"), "app": app.get("version")}


class Sampler:
    """Samples the phone's own telemetry on an interval, so every speed row carries the thermal state and memory
    footprint it was measured under. Board energy has no counterpart on a phone, so it stays unreported."""

    def __init__(self, device, interval_ms=1000):
        self.device = device
        self.interval = interval_ms / 1000
        self.samples = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while not self.stopped.wait(self.interval):
            health = self.device.health()
            if health is None:
                continue
            telemetry = health.get("telemetry") or {}
            self.samples.append((time.time(), telemetry))

    def window(self, start, end):
        inside = [telemetry for stamp, telemetry in self.samples if start <= stamp <= end]
        if not inside:
            return {}
        states = [entry.get("thermal_state") for entry in inside if entry.get("thermal_state")]
        footprints = [entry.get("footprint_mib") for entry in inside if entry.get("footprint_mib")]
        levels = [entry.get("battery_level") for entry in inside if entry.get("battery_level") is not None]
        return {
            "thermal": max(states, key=lambda state: THERMAL_ORDER.index(state)) if states else None,
            "thermal_start": states[0] if states else None,
            "footprint_mib": round(max(footprints)) if footprints else None,
            "battery": round(statistics.mean(levels), 3) if levels else None,
        }

    def energy_wh(self, start, end):
        return None

    def close(self):
        self.stopped.set()


class Guard(threading.Thread):
    """Stops a run when the app on the phone stops answering — it was backgrounded, jetsammed or the cable came
    out — instead of recording the missing answers as wrong ones."""

    def __init__(self, device, halt, events, interval=15):
        super().__init__(daemon=True)
        self.device = device
        self.halt = halt
        self.events = events
        self.interval = interval
        self.stopped = threading.Event()
        self.fault = None

    def run(self):
        misses = 0
        while not self.stopped.wait(self.interval):
            misses = misses + 1 if not self.device.reachable() else 0
            if misses >= 2:
                self.fault = f"mbenchd at {self.device.control_url} stopped answering"
                self.events.emit("abort", reason=self.fault)
                self.halt.set("device")
                return

    def stop(self):
        self.stopped.set()


def tool():
    """pymobiledevice3 is what carries files and ports over the cable; it is not a dependency of mbench itself."""
    found = os.environ.get("MBENCH_PMD") or shutil.which("pymobiledevice3")
    if not found:
        raise RuntimeError("pymobiledevice3 is not installed; `pipx install pymobiledevice3`, or set MBENCH_PMD")
    return found


def run_tool(*args, timeout=600, capture=True):
    return subprocess.run([tool(), *args], capture_output=capture, text=True, timeout=timeout)


def bundle_id():
    """The app's id on the device. xtool installs under an XTL- prefix of its own, so the id is looked up rather
    than assumed."""
    override = os.environ.get("MBENCH_BUNDLE_ID")
    if override:
        return override
    completed = run_tool("apps", "list")
    if completed.returncode != 0:
        raise RuntimeError(f"could not list apps on the phone: {completed.stderr.strip()[:200]}")
    found = [key for key in json.loads(completed.stdout) if key.endswith(BUNDLE_SUFFIX)]
    if not found:
        raise RuntimeError(f"no app ending in {BUNDLE_SUFFIX} is installed; build and install ios/mbenchd first")
    return sorted(found, key=len)[0]


def push(path, name=None):
    """Copies a .gguf into the app's Documents/models over USB."""
    target = f"Documents/models/{name or path.name}"
    completed = run_tool("apps", "push", bundle_id(), str(path), target, timeout=7200)
    if completed.returncode != 0:
        raise RuntimeError(f"push failed: {completed.stderr.strip()[:300]}")
    return target


def pull(remote, local):
    completed = run_tool("apps", "pull", bundle_id(), remote, str(local))
    if completed.returncode != 0:
        raise RuntimeError(f"pull failed: {completed.stderr.strip()[:300]}")
    return local


def forward(local_port, device_port):
    """One port forward, left running until it is killed; the caller keeps the process."""
    return subprocess.Popen([tool(), "usbmux", "forward", str(local_port), str(device_port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
