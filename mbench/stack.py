import hashlib
import json
import re
import subprocess
from importlib import metadata
from pathlib import Path

from . import __version__ as VERSION
from . import gpu, paths

PLACEHOLDER_VERSIONS = ("", "0.0.0", "0.0.0.dev0", "unknown", "dev")
ENGINE_LABELS = {"sglang": "SGLang", "llama.cpp": "llama.cpp", "vllm": "vLLM"}
ENGINE_PACKAGES = {"sglang": "sglang", "vllm": "vllm"}


def server_pids():
    return [app["pid"] for app in gpu.compute_apps() if gpu.is_server(app["pid"])]


def venv_of(pid):
    """The virtualenv a running server was started from: its own argv names it, while /proc/<pid>/exe resolves past the
    virtualenv to the interpreter it links to."""
    for token in gpu.cmdline(pid).split():
        path = Path(token)
        if path.is_absolute() and path.parent.name == "bin" and (path.parent.parent / "pyvenv.cfg").exists():
            return path.parent.parent
    return None


def dist_infos(venv, package):
    return sorted(venv.glob(f"lib/python3*/site-packages/{package}-*.dist-info"))


def editable_checkout(venv, package):
    """Where an editable install points: a source build carries no version of its own, so its checkout names it."""
    for info in dist_infos(venv, package):
        try:
            data = json.loads((info / "direct_url.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        url = data.get("url") or ""
        if (data.get("dir_info") or {}).get("editable") and url.startswith("file://"):
            return Path(url.removeprefix("file://"))
    return None


def installed_version(venv, package):
    for info in dist_infos(venv, package):
        match = re.fullmatch(re.escape(package) + r"-(.+)\.dist-info", info.name)
        if match and match.group(1) not in PLACEHOLDER_VERSIONS:
            return match.group(1)
    return None


def git_head(directory):
    completed = subprocess.run(["git", "-C", str(directory), "log", "-1", "--format=%h %cs"],
                               capture_output=True, text=True)
    parts = completed.stdout.split()
    return {"commit": parts[0], "dated": parts[1]} if completed.returncode == 0 and len(parts) == 2 else None


def venv_build(engine):
    """What the server is running from when it reports no version of its own: the checkout of a source install, or
    the version its virtualenv has on disk."""
    package = ENGINE_PACKAGES.get(engine)
    if not package:
        return {}
    for pid in server_pids():
        venv = venv_of(pid)
        if not venv:
            continue
        checkout = editable_checkout(venv, package)
        head = git_head(checkout) if checkout else None
        if head:
            return {**head, "source": str(checkout)}
        version = installed_version(venv, package)
        if version:
            return {"version": version}
    return {}


def reported(info):
    payload = (info or {}).get("info") or {}
    version = str(payload.get("version") or "").strip()
    build = str(payload.get("build_info") or "").strip()
    found = {}
    if version and version not in PLACEHOLDER_VERSIONS:
        found["version"] = version
    if build:
        found["build"] = build
    return found


def build(engine, info):
    """The build of the server that answered: what it reports about itself, or what its virtualenv says it runs."""
    found = reported(info)
    return {"engine": engine, **(found or venv_build(engine))}


def as_build(found):
    """Runs from before the engine was recorded as a whole kept llama.cpp's build string on its own."""
    if isinstance(found, str):
        return {"build": found} if found else {}
    return found or {}


def build_id(found):
    """What names the build itself, without the engine in front of it, so an older run's bare string still compares."""
    found = as_build(found)
    return found.get("version") or found.get("build") or found.get("commit")


def build_label(found):
    found = as_build(found)
    engine = ENGINE_LABELS.get(found.get("engine"), found.get("engine"))
    return " ".join(part for part in (engine, build_id(found)) if part) or None


def host_key(hardware):
    hardware = hardware or {}
    if hardware.get("class") == "phone":
        return f"{hardware.get('device') or '?'}|{hardware.get('os') or '?'}"
    return f"{hardware.get('gpu') or '?'}|{hardware.get('driver') or '?'}"


def host_label(hardware):
    hardware = hardware or {}
    if hardware.get("class") == "phone":
        parts = [hardware.get("device") or "unknown phone", hardware.get("soc"), hardware.get("os")]
        return " · ".join(part for part in parts if part)
    parts = [hardware.get("gpu") or "unknown GPU"]
    if hardware.get("driver"):
        parts.append(f"driver {hardware['driver']}")
    return " · ".join(parts)


def of(run):
    """A run's stack: the card and driver that answered, and the build of the server in front of them. Quality compares
    across stacks; speed, throughput and energy only compare within one."""
    hardware = (run or {}).get("hardware") or {}
    found = as_build(((run or {}).get("server") or {}).get("build"))
    return {"host": host_key(hardware), "hostLabel": host_label(hardware), "gpu": hardware.get("gpu"),
            "driver": hardware.get("driver"), "vramMib": hardware.get("vram_mib"),
            "deviceClass": hardware.get("class") or "gpu", "device": hardware.get("device"),
            "soc": hardware.get("soc"), "bandwidthGbs": hardware.get("bandwidth_gbs"), "ramGb": hardware.get("ram_gb"),
            "engine": found.get("engine"), "build": build_label(found), "source": found.get("source")}


def changed(previous, hardware, found):
    """What moved since a model's last run; a speed number only carries over while these hold still."""
    moved = []
    was_hardware = (previous or {}).get("hardware") or {}
    was_build = ((previous or {}).get("server") or {}).get("build")
    if was_hardware and host_key(was_hardware) != host_key(hardware):
        moved.append(f"{host_label(was_hardware)} → {host_label(hardware)}")
    was_id, now_id = build_id(was_build), build_id(found)
    if was_id and now_id and was_id != now_id:
        moved.append(f"{build_label(was_build)} → {build_label(found)}")
    return moved


def checkout_commit(root):
    """The commit of the source checkout mbench runs from (an editable install), marked -dirty when its package files changed."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()

    try:
        top = git("rev-parse", "--show-toplevel")
        if not top or Path(top).resolve() != root.resolve():
            return None
        return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain", "--", "mbench") else "")
    except FileNotFoundError:
        return None


def installed_commit():
    """The commit `uv tool install git+…` built from, which uv and pip record in the package's direct_url.json (PEP 610)."""
    try:
        record = json.loads(metadata.distribution("mbench").read_text("direct_url.json") or "{}")
    except metadata.PackageNotFoundError:
        return None
    return ((record.get("vcs_info") or {}).get("commit_id") or "")[:7] or None


def harness():
    """The mbench that is measuring: a run records it when it starts, not when it was queued, because the code can move
    between the two."""
    return f"{VERSION}+{checkout_commit(paths.PACKAGE.parent) or installed_commit() or 'nogit'}"


def digest_of(path):
    """The sha256 of a weights file, cached against its size and modification time. It is what lets a phone run and a
    desktop run prove they scored the same artifact rather than two files with the same name."""
    try:
        found = Path(path)
        stat = found.stat()
    except (OSError, TypeError):
        return None
    cache_file = paths.CACHE / "digests.json"
    key = f"{found}:{stat.st_size}:{stat.st_mtime_ns}"
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, json.JSONDecodeError):
        cache = {}
    if key in cache:
        return cache[key]
    digest = hashlib.sha256()
    with found.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    cache[key] = digest.hexdigest()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache))
    return cache[key]
