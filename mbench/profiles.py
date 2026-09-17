import hashlib
import json
import shlex
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from . import paths, phone


@dataclass
class Profile:
    id: str
    name: str
    engine: str
    thinking: str
    context: int | None
    efforts: list = field(default_factory=list)
    hf_id: str | None = None
    quantization: str | None = None
    spec: dict = field(default_factory=dict)
    engine_meta: dict = field(default_factory=dict)
    cmd: str = ""
    fingerprint: str = ""
    sources: dict = field(default_factory=dict)
    phone: dict = field(default_factory=dict)
    remote: dict = field(default_factory=dict)
    base_url: str | None = None
    twin: str | None = None
    served_as: str | None = None

    @property
    def served(self):
        """The name the server answering for this model knows it by. A llama-swap model is its own id; a server
        somewhere else on the network was started with a path or a repo id of its own."""
        return self.served_as or self.id

    def to_dict(self):
        return asdict(self)


def swap_config():
    return yaml.safe_load(paths.LLAMA_SWAP_CONFIG.read_text()) or {}


def expand(cmd, macros):
    for key, value in macros.items():
        cmd = cmd.replace("${" + key + "}", " ".join(str(value).split()))
    return " ".join(cmd.split())


def omp_overrides():
    """llama-swap model overrides from Oh My Pi's models.yml; they already say how each model takes a thinking level."""
    try:
        data = yaml.safe_load(paths.OMP_MODELS.read_text()) or {}
    except OSError:
        return {}
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "modelOverrides" and isinstance(value, dict):
                    found.update(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return found


def user_profiles():
    try:
        return tomllib.loads(paths.PROFILES.read_text())
    except OSError:
        return {}


def thinking_style(format_name):
    if not format_name:
        return "none"
    if format_name == "openai":
        return "openai"
    if format_name.startswith("qwen"):
        return "qwen"
    return "none"


def launcher_scripts(cmd):
    """Small files the command runs (launcher scripts), skipping system binaries and multi-GB model files."""
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()
    found = []
    for token in tokens:
        path = Path(token)
        if path.is_absolute() and not token.startswith(("/usr/", "/bin/")) and path.is_file() and path.stat().st_size < 1_000_000:
            found.append(path)
    return found


def fingerprint(cmd):
    """Hashes the launch command plus the scripts it runs, so editing a launcher counts as a new configuration."""
    digest = hashlib.sha256(cmd.encode())
    for path in launcher_scripts(cmd):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def engine_of(cmd):
    """Reads the engine off the command, or off the launcher script when the command only runs a wrapper."""
    haystacks = [cmd] + [path.read_text(errors="replace") for path in launcher_scripts(cmd)]
    for needle, engine in (("llama-server", "llama.cpp"), ("sglang", "sglang"), ("vllm", "vllm")):
        if any(needle in haystack for haystack in haystacks):
            return engine
    return "other"


def phone_request(model_id, settings):
    """What mbenchd is asked to load: the phone table minus the addresses, which say where to send it rather than what
    to run. `file` names the .gguf on the phone when it differs from the model id."""
    request = {key: value for key, value in settings.items() if key not in ("control_url", "server_url")}
    request["model"] = request.pop("file", None) or request.get("model") or model_id
    return {key: value for key, value in request.items() if value is not None}


def phone_profile(model_id, user):
    """A model that runs on a phone has no llama-swap entry: models.toml says everything, and the load request sent
    to mbenchd stands in for the launch command, so changing the context or the KV type counts as a new configuration."""
    settings = dict(user["phone"])
    request = phone_request(model_id, settings)
    body = json.dumps(request, sort_keys=True)
    return Profile(
        id=model_id,
        name=user.get("name") or model_id,
        engine=user.get("engine_kind") or "llama.cpp",
        thinking=user.get("thinking") or "none",
        context=user.get("context") or request.get("n_ctx"),
        efforts=list(user.get("efforts") or []),
        hf_id=user.get("hf_id"),
        quantization=user.get("quantization"),
        spec=user.get("spec") or {},
        engine_meta=user.get("engine") or {},
        cmd=body,
        fingerprint=hashlib.sha256(body.encode()).hexdigest()[:16],
        sources={key: "models.toml" for key in ("name", "thinking", "context", "efforts", "hf_id")},
        phone=settings,
        twin=user.get("twin"),
        base_url=settings.get("server_url") or phone.SERVER_URL,
    )


def remote_profile(model_id, user):
    """A model served by an OpenAI-compatible server on another machine — a Mac running MLX, say. There is no
    llama-swap entry and nothing to launch: models.toml says where the server is and what it calls the model."""
    settings = dict(user["remote"])
    body = json.dumps({key: value for key, value in sorted(settings.items()) if key != "base_url"}, sort_keys=True)
    return Profile(
        id=model_id,
        name=user.get("name") or model_id,
        engine=user.get("engine_kind") or "mlx",
        thinking=user.get("thinking") or "none",
        context=user.get("context"),
        efforts=list(user.get("efforts") or []),
        hf_id=user.get("hf_id"),
        quantization=user.get("quantization"),
        spec=user.get("spec") or {},
        engine_meta=user.get("engine") or {},
        cmd=body,
        fingerprint=hashlib.sha256(body.encode()).hexdigest()[:16],
        sources={key: "models.toml" for key in ("name", "thinking", "context", "efforts", "hf_id")},
        remote=settings,
        twin=user.get("twin"),
        base_url=settings.get("base_url"),
        served_as=settings.get("model"),
    )


def resolve(model_id):
    declared = user_profiles().get(model_id, {})
    if declared.get("phone"):
        return phone_profile(model_id, declared)
    if declared.get("remote"):
        return remote_profile(model_id, declared)
    config = swap_config()
    models = config.get("models") or {}
    entry = models.get(model_id)
    if entry is None:
        owner = next((key for key, value in models.items() if model_id in (value.get("aliases") or [])), None)
        if owner is None:
            known = ", ".join(sorted(models))
            raise KeyError(f"{model_id} is not a llama-swap model. Known: {known}")
        model_id, entry = owner, models[owner]
    cmd = expand(str(entry.get("cmd", "")), config.get("macros") or {})
    omp = omp_overrides().get(model_id, {})
    user = user_profiles().get(model_id, {})
    omp_format = (omp.get("compat") or {}).get("thinkingFormat")
    sources = {
        "name": "models.toml" if user.get("name") else "llama-swap",
        "thinking": "models.toml" if user.get("thinking") else (f"omp thinkingFormat={omp_format}" if omp_format else "default"),
        "context": "models.toml" if user.get("context") else ("omp contextWindow" if omp.get("contextWindow") else "server"),
        "efforts": "models.toml" if user.get("efforts") else ("omp thinking.efforts" if (omp.get("thinking") or {}).get("efforts") else "none declared"),
        "hf_id": "models.toml" if user.get("hf_id") else "missing",
    }
    return Profile(
        id=model_id,
        name=user.get("name") or entry.get("name") or model_id,
        engine=user.get("engine_kind") or engine_of(cmd),
        thinking=user.get("thinking") or thinking_style(omp_format),
        context=user.get("context") or omp.get("contextWindow"),
        efforts=list(user.get("efforts") or (omp.get("thinking") or {}).get("efforts") or []),
        hf_id=user.get("hf_id"),
        quantization=user.get("quantization"),
        spec=user.get("spec") or {},
        engine_meta=user.get("engine") or {},
        cmd=cmd,
        fingerprint=fingerprint(cmd),
        sources=sources,
    )


def group_of(model_id, config=None):
    """The llama-swap group a model belongs to, when it is in one that keeps its members loaded together."""
    config = config if config is not None else swap_config()
    for name, group in (config.get("groups") or {}).items():
        if model_id in (group.get("members") or []) and group.get("swap") is False:
            return name
    return None
