import http.client
import json
import os
from pathlib import Path
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
SETTLE_S = 5
LOAD_ATTEMPTS = 3

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
COOLDOWN_FLOOR_S = 60
COOLDOWN_CAP_S = 900
COOLDOWN_POLL_S = 10
COOLDOWN_HOLD_S = 60


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

    def answering(self):
        """Whether the model server itself is alive, not just the app that started it. A wedged llama-server accepts
        the connection and never replies, which the control port reports as a healthy run for as long as the request
        hangs — which is forever."""
        health = self.health()
        if health is None:
            return False
        if ((health.get("server") or {}).get("state")) != "running":
            return True
        try:
            with urllib.request.urlopen(self.server_url + "/health", timeout=HEALTH_TIMEOUT_S) as response:
                return response.status == 200
        except (OSError, urllib.error.HTTPError, http.client.HTTPException):
            return False

    def models(self):
        return (self.get("/mb/models") or {}).get("data") or []

    def load(self, request):
        """Loads the model and waits for llama-server to answer; the phone needs minutes for a first load, since
        Metal compiles every kernel from source on the device. A dropped connection is retried: the app may still be
        tearing down the model the run before it used, and a lost connection is not a lost phone."""
        for attempt in range(LOAD_ATTEMPTS):
            try:
                return self.post("/mb/load", request, timeout=LOAD_TIMEOUT_S)
            except (OSError, urllib.error.HTTPError, http.client.HTTPException) as error:
                if attempt == LOAD_ATTEMPTS - 1 or not self.reachable():
                    raise Unreachable(f"{self.control_url} did not load the model: {error!r}") from error
                time.sleep(SETTLE_S)

    def unload(self):
        """Asks the app to stop the server and waits for it to say it has, so the next load does not race the
        teardown."""
        try:
            self.post("/mb/unload", {}, timeout=300)
        except (OSError, urllib.error.HTTPError, http.client.HTTPException):
            pass
        for _ in range(60):
            state = ((self.health() or {}).get("server") or {}).get("state")
            if state in (None, "idle", "failed"):
                return
            time.sleep(1)

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

    def cooldown(self, target="nominal", floor=COOLDOWN_FLOOR_S, cap=COOLDOWN_CAP_S, hold=COOLDOWN_HOLD_S, report=None):
        """Waits for the phone to come back to its cold state, and to stay there, before a measurement starts. Without
        this, every run after the first is measured on heat the run before it made, and a queue reads as if the models
        got worse down the list. The device has to report `target` for `hold` seconds together — the readable stand-in
        for the published rule of a temperature that has stopped moving — after idling at least `floor`, giving up at
        `cap` and saying it gave up."""
        started = time.time()
        entry = ((self.health() or {}).get("telemetry") or {}).get("thermal_state")
        cool_since = None
        while True:
            waited = time.time() - started
            telemetry = (self.health() or {}).get("telemetry") or {}
            state = telemetry.get("thermal_state")
            cool = state is not None and THERMAL_ORDER.index(state) <= THERMAL_ORDER.index(target)
            cool_since = (cool_since or time.time()) if cool else None
            settled = cool_since is not None and time.time() - cool_since >= hold
            if waited >= cap or (waited >= floor and settled):
                return {"waited_s": round(waited, 1), "started": round(started, 1), "ended": round(time.time(), 1),
                        "entry": entry, "exit": state, "reached": bool(settled),
                        "battery": telemetry.get("battery_level"), "battery_state": telemetry.get("battery_state")}
            if report and int(waited) % 60 < COOLDOWN_POLL_S:
                report(f"cooling: {state} after {int(waited)}s")
            time.sleep(COOLDOWN_POLL_S)

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

    def timeline(self):
        """Every sample the run was measured under, so the board can show when the phone got hot and what the
        answers cost the battery."""
        return [{"t": round(stamp, 1), "thermal": entry.get("thermal_state"),
                 "footprint_mib": entry.get("footprint_mib"), "available_mib": entry.get("available_mib"),
                 "battery": entry.get("battery_level"), "battery_state": entry.get("battery_state")}
                for stamp, entry in self.samples]

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
        seen = None
        while not self.stopped.wait(self.interval):
            health = self.device.health()
            uptime = ((health or {}).get("telemetry") or {}).get("uptime")
            if uptime is not None and seen is not None and uptime < seen:
                self.fault = ("iOS killed the app and it relaunched: the model needed more memory than the phone "
                              "would give it")
                self.events.emit("abort", reason=self.fault)
                self.halt.set("memory")
                return
            seen = uptime if uptime is not None else seen
            misses = misses + 1 if not self.device.answering() else 0
            if misses >= 4:
                self.fault = (f"mbenchd at {self.device.control_url} stopped answering"
                              if health is None else
                              f"the model server at {self.device.server_url} stopped answering while the app kept running")
                self.events.emit("abort", reason=self.fault)
                self.halt.set("device")
                return

    def stop(self):
        self.stopped.set()


TOOL_PATHS = ("~/.local/bin/pymobiledevice3", "~/.local/share/pmd-venv/bin/pymobiledevice3",
              "~/.local/pipx/venvs/pymobiledevice3/bin/pymobiledevice3", "/usr/local/bin/pymobiledevice3")


def tool():
    """pymobiledevice3 carries files and ports over the cable; it is not a dependency of mbench itself. A worker runs
    as a systemd unit, whose PATH is the system's and not the shell's, so the usual places are searched by hand
    rather than trusted to `which`."""
    found = os.environ.get("MBENCH_PMD") or shutil.which("pymobiledevice3")
    if not found:
        found = next((str(path) for candidate in TOOL_PATHS
                      if (path := Path(candidate).expanduser()).exists()), None)
    if not found:
        raise RuntimeError("pymobiledevice3 is not installed; `pipx install pymobiledevice3`, or set MBENCH_PMD")
    return found


def run_tool(*args, timeout=600, capture=True):
    return subprocess.run([tool(), *args], capture_output=capture, text=True, timeout=timeout)


def complain(completed):
    """pymobiledevice3 writes a timestamped log line rather than a message; only the part after its level is worth
    repeating, and its usual failure means the cable, not the tool."""
    text = ((completed.stderr or "") + (completed.stdout or "")).strip()
    said = text.rsplit("ERROR ", 1)[-1].splitlines()[0] if text else ""
    if "usbmuxd" in said:
        return "no phone on the cable — plug it in, unlock it, and trust this computer"
    return said[:200] or "pymobiledevice3 said nothing"


def apps():
    """Every mbenchd on the device, keyed by bundle id. xtool installs under an XTL- prefix of its own, so the id is
    looked up rather than assumed."""
    completed = run_tool("apps", "list")
    if completed.returncode != 0:
        raise RuntimeError(f"could not list apps on the phone: {complain(completed)}")
    listed = json.loads(completed.stdout)
    found = sorted((key for key in listed if key.endswith(BUNDLE_SUFFIX)), key=len)
    if not found:
        raise RuntimeError(f"no app ending in {BUNDLE_SUFFIX} is installed; build and install ios/mbenchd first")
    return found, listed


def bundle_id():
    return os.environ.get("MBENCH_BUNDLE_ID") or apps()[0][0]


def installed():
    """What the phone actually holds: the bundle it is installed under, its name and its version. A build that
    never reached the device is the difference between a fixed bug and a bug reported twice."""
    found, listed = apps()
    entry = listed[found[0]] or {}
    return {"bundle": found[0], "name": entry.get("CFBundleDisplayName") or entry.get("CFBundleName"),
            "version": entry.get("CFBundleShortVersionString"), "build": entry.get("CFBundleVersion")}


def launch(kill_existing=True):
    """Starts the app from the host. xtool's launch needs a debugserver it cannot attach to on iOS 17+, so this goes
    through DVT process control, which pymobiledevice3 reaches over a userspace tunnel without root."""
    args = ["developer", "dvt", "launch"] + (["--kill-existing"] if kill_existing else []) + [bundle_id()]
    completed = run_tool(*args, timeout=300)
    text = (completed.stdout or "") + (completed.stderr or "")
    if "launched with pid" not in text:
        raise RuntimeError(f"could not launch the app: {text.strip()[-300:]}")
    return int(text.rsplit("pid", 1)[-1].split()[0])


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
