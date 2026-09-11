import json
import re
import time
import urllib.error
import urllib.request

from . import paths

SERVER_INFO_KEYS = (
    "model_path", "served_model_name", "context_length", "attention_backend", "prefill_attention_backend",
    "decode_attention_backend", "mem_fraction_static", "max_running_requests", "chunked_prefill_size", "kv_cache_dtype",
    "quantization", "dtype", "tp_size", "page_size", "sampling_backend", "moe_runner_backend", "speculative_algorithm",
    "speculative_draft_model_path", "speculative_num_draft_tokens", "speculative_draft_window_size",
)


def get(path, timeout=10):
    with urllib.request.urlopen(paths.SWAP_URL + path, timeout=timeout) as response:
        return response.read()


def reachable():
    try:
        get("/v1/models", timeout=5)
        return True
    except OSError:
        return False


def running():
    return json.loads(get("/running")).get("running", [])


def ensure_loaded(model, timeout=1800):
    """Sends a tiny request so llama-swap swaps the model in; returns seconds until it answered."""
    started = time.time()
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 16}).encode()
    request = urllib.request.Request(paths.SWAP_URL + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
    return round(time.time() - started, 1)


def unload():
    try:
        get("/unload", timeout=60)
    except OSError:
        pass


def upstream(model, path, timeout=10):
    try:
        return get(f"/upstream/{model}{path}", timeout=timeout)
    except (OSError, urllib.error.HTTPError):
        return None


def server_info(model):
    """SGLang answers /get_server_info, llama.cpp answers /props; either tells what the server really runs with."""
    for path in ("/get_server_info", "/props"):
        raw = upstream(model, path)
        if not raw:
            continue
        try:
            return {"endpoint": path, "info": json.loads(raw)}
        except json.JSONDecodeError:
            continue
    return {}


def trimmed_info(info):
    payload = info.get("info") or {}
    if info.get("endpoint") == "/get_server_info":
        return {key: payload.get(key) for key in SERVER_INFO_KEYS if payload.get(key) is not None}
    settings = payload.get("default_generation_settings") or {}
    return {
        "build": payload.get("build_info"),
        "model_path": payload.get("model_path"),
        "n_ctx": settings.get("n_ctx"),
        "total_slots": payload.get("total_slots"),
    }


def context_from(info):
    payload = info.get("info") or {}
    if info.get("endpoint") == "/get_server_info":
        return payload.get("context_length")
    return (payload.get("default_generation_settings") or {}).get("n_ctx")


def spec_counters(model):
    """Generated-token and verify-pass counters from SGLang's Prometheus endpoint, summed over label sets."""
    raw = upstream(model, "/metrics")
    if not raw:
        return None
    text = raw.decode(errors="replace")
    values = {}
    for name in ("sglang:generation_tokens_total", "sglang:spec_verify_calls_total"):
        series = re.findall(rf"^{re.escape(name)}\{{[^}}]*\}} ([0-9.eE+]+)$", text, flags=re.MULTILINE)
        values[name] = sum(float(value) for value in series) if series else None
    return values


def accept_length(before, after):
    if not before or not after:
        return None
    generated = (after["sglang:generation_tokens_total"] or 0) - (before["sglang:generation_tokens_total"] or 0)
    verifies = (after["sglang:spec_verify_calls_total"] or 0) - (before["sglang:spec_verify_calls_total"] or 0)
    return round(generated / verifies, 3) if verifies > 0 else None
