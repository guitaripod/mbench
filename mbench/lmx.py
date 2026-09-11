import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import __version__, gpu, paths, suite, swap
from .engine import injection
from .shim import start as start_shim

LMX = shutil.which("lmx") or str(paths.HOME / ".local/bin/lmx")
LMX_CONFIG = paths.HOME / ".config/localmaxxing/config.json"
API = "https://www.localmaxxing.com/api"
LMX_ENGINES = {"sglang": "sglang", "llama.cpp": "llama.cpp", "vllm": "vllm"}
SUBMIT_CHOICES = ("all", "speed", "evals")
PAYLOAD_FIELDS = (
    "hfId", "hardware", "engineName", "quantization", "tokSOut", "tokSPrefill", "tokSTotal", "ttftMs", "backend",
    "batchSize", "engineCommit", "engineFlags", "engineRepository", "engineVersion", "gpuPowerWatts", "modelRevision",
    "notes", "outputTokens", "peakVramGb", "prefillTokens", "promptTokens",
)


def available():
    return Path(LMX).exists() and LMX_CONFIG.exists()


def missing_fields(profile):
    return [name for name in ("hf_id", "quantization") if not getattr(profile, name)]


def hardware_file():
    if paths.HARDWARE.exists():
        return paths.HARDWARE
    paths.CONFIG.mkdir(parents=True, exist_ok=True)
    subprocess.run([LMX, "hardware", "--out", str(paths.HARDWARE)], check=True, capture_output=True)
    return paths.HARDWARE


def spec_flags(profile):
    spec = profile.spec or {}
    if not spec.get("method"):
        return []
    flags = ["--spec-decoding", "--spec-method", spec["method"]]
    if spec.get("draft"):
        flags += ["--spec-draft-model", spec["draft"]]
    if spec.get("tokens_per_step"):
        flags += ["--spec-num-tokens", str(spec["tokens_per_step"])]
    if spec.get("window"):
        flags += ["--spec-draft-window-size", str(spec["window"])]
    return flags


def spec_counts(profile, output_tokens, length):
    """Scales measured acceptance onto one reported request so accepted + drafts/step == outputTokens, as the site checks."""
    per_step = (profile.spec or {}).get("tokens_per_step")
    if not per_step or not length:
        return {}
    steps = output_tokens / length
    return {"specMeanAcceptedLength": round(length, 3), "specAcceptanceRate": round((length - 1) / per_step, 4),
            "specDraftTokens": round(steps * per_step), "specAcceptedTokens": round(output_tokens - steps)}


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def evidence(run):
    """What the verified-run check needs and the lmx client never sends: prompt hash and sample, the output, per-iteration timings."""
    prompt = run.get("prompt") or ""
    output = run.get("outputText") or ""
    return {
        "promptSha256": sha256(prompt),
        "promptSample": prompt[:2000],
        "outputSha256": sha256(output),
        "outputSample": output if len(output) <= 4000 else output[:2996] + "\n…\n" + output[-1000:],
        "engineTimingsRaw": {"source": f"mbench {__version__}: OpenAI-compatible usage and client stream timings per timed iteration",
                             "samples": run.get("samples", [])},
    }


def payload(run, profile):
    body = {key: run[key] for key in PAYLOAD_FIELDS if run.get(key) is not None}
    if profile.context:
        body["contextLength"] = profile.context
    body.update(evidence(run))
    return body


def api(path, body):
    """Posts to the localmaxxing API with the key lmx saved, bypassing the lmx client, which drops the verification evidence."""
    key = json.loads(LMX_CONFIG.read_text())["apiKey"]
    request = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": f"mbench/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    try:
        parsed = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        return status, {"error": raw[:300].decode(errors="replace")}
    return status, parsed if isinstance(parsed, dict) else {"body": parsed}


def created(response):
    for candidate in (response, response.get("speedTest"), response.get("run"), response.get("data")):
        if isinstance(candidate, dict) and candidate.get("id"):
            return candidate["id"], candidate.get("verifiedRun")
    return None, None


def speed_runs(profile, run_dir, run_id, log):
    """lmx measures the two canonical prompts; mbench submits them itself, with evidence, so they can earn the Verified badge."""
    engine = LMX_ENGINES.get(profile.engine)
    if not engine:
        log(f"localmaxxing: no speed-test engine for {profile.engine}, skipped")
        return []
    folder = run_dir / "lmx"
    folder.mkdir(parents=True, exist_ok=True)
    prompts = suite.canonical_prompts()
    sampler = gpu.Sampler()
    submitted = []
    try:
        for prompt_id in ("code-v1", "reasoning-v1"):
            prompt_file = folder / f"prompt-{prompt_id}.txt"
            prompt_file.write_text(prompts[prompt_id])
            out = folder / f"speed-{prompt_id}.json"
            command = [
                LMX, "speed-test", "run", engine, "--mode", "remote", "--base-url", paths.SWAP_URL,
                "--served-model", profile.id, "--hf-id", profile.hf_id, "--quantization", profile.quantization,
                "--hardware", str(hardware_file()), "--prompt-file", str(prompt_file), "--max-tokens", "512",
                "--warmup", "1", "--iterations", "3", "--backend", "cuda", "--runs-dir", str(folder / "runs"),
                "--out", str(out), *spec_flags(profile),
            ]
            before = swap.spec_counters(profile.id) if profile.engine == "sglang" else None
            started = time.time()
            measured = subprocess.run(command, capture_output=True, text=True)
            window = sampler.window(started, time.time())
            if measured.returncode != 0 or not out.exists():
                log(f"localmaxxing: speed run {prompt_id} failed: {(measured.stdout + measured.stderr)[-400:]}")
                continue
            run = json.loads(out.read_text())
            flags = run.setdefault("engineFlags", {})
            flags["commandSnippet"] = profile.cmd[:3900]
            flags.update(spec_counts(profile, run.get("outputTokens") or 512,
                                     swap.accept_length(before, swap.spec_counters(profile.id)) if before else None))
            meta = profile.engine_meta or {}
            run.update({key: value for key, value in (("engineRepository", meta.get("repository")),
                                                       ("engineCommit", meta.get("commit")),
                                                       ("engineVersion", meta.get("version"))) if value})
            if window.get("vram_mib"):
                run["peakVramGb"] = round(window["vram_mib"] / 1024, 1)
            if window.get("power_w"):
                run["gpuPowerWatts"] = [window["power_w"]]
            run["notes"] = (f"mbench {__version__}, run {run_id}: canonical prompt, single request, greedy, "
                            "the server's default reasoning effort.")
            body = payload(run, profile)
            out.write_text(json.dumps({**run, "apiPayload": body}, indent=2))
            status, check = api("/speed-tests/dry-run", body)
            if status != 200 or not check.get("valid"):
                log(f"localmaxxing: speed run {prompt_id} rejected by dry-run ({status}): {json.dumps(check)[:300]}")
                continue
            found = check.get("verification")
            verification = found if isinstance(found, dict) else {}
            status, response = api("/speed-tests", body)
            remote_id, verified = created(response)
            verified = verification.get("verified") if verified is None else verified
            submitted.append({"kind": f"speed.{prompt_id}", "remote_id": remote_id, "value": run.get("tokSOut"),
                              "ok": status in (200, 201) and remote_id is not None, "verified": verified,
                              "issues": verification.get("issues") or []})
            log(f"localmaxxing: speed {prompt_id} {run.get('tokSOut')} tok/s -> {remote_id or f'HTTP {status}'}"
                f"{' (verified)' if verified else ''}")
    finally:
        sampler.close()
    return submitted


def parse_shards(text):
    entries = []
    for block in text.split("[localmaxxing] eval_shard_submitted")[1:]:
        accuracy = re.search(r"accuracyPct: ([0-9.]+)", block)
        index = re.search(r"shardIndex: (\d+)", block)
        if accuracy and index:
            entries.append({"shard": int(index.group(1)), "accuracy": float(accuracy.group(1))})
    return entries


def shard_runs(profile, effort, run_dir, log):
    """GSM8K and HellaSwag through the shim, which gives GSM8K the run's effort and gets HellaSwag scored on SGLang.
    --all-missing means shards the site already has for this model and quantization are not sent again."""
    folder = run_dir / "lmx"
    folder.mkdir(parents=True, exist_ok=True)
    server, port = start_shim(paths.SWAP_URL, injection(profile, effort))
    results = {}
    try:
        for dataset in ("gsm8k", "hellaswag"):
            command = [
                LMX, "eval", "shard", dataset, "--base-url", f"http://127.0.0.1:{port}", "--model", profile.hf_id,
                "--served-model", profile.id, "--hardware", str(hardware_file()), "--quantization", profile.quantization,
                "--thinking-level", effort, "--max-tokens", "16384", "--concurrency", str(suite.CONCURRENCY),
                "--all-missing", "--submit",
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            text = completed.stdout + completed.stderr
            (folder / f"shards-{dataset}.log").write_text(text)
            results[dataset] = parse_shards(text)
            log(f"localmaxxing: {dataset} submitted {len(results[dataset])} new shards")
    finally:
        server.shutdown()
    return results
