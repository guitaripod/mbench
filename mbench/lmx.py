import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import __version__, gpu, paths, suite, swap
from .shim import start as start_shim

LMX = shutil.which("lmx") or str(paths.HOME / ".local/bin/lmx")
LMX_ENGINES = {"sglang": "sglang", "llama.cpp": "llama.cpp", "vllm": "vllm"}


def available():
    return Path(LMX).exists()


def missing_fields(profile):
    return [name for name in ("hf_id", "quantization") if not getattr(profile, name)]


def hardware_file():
    if paths.HARDWARE.exists():
        return paths.HARDWARE
    paths.CONFIG.mkdir(parents=True, exist_ok=True)
    reviewed = paths.HOME / ".omp-bridge/workdir/hardware.json"
    if reviewed.exists():
        shutil.copyfile(reviewed, paths.HARDWARE)
    else:
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


def speed_runs(profile, run_dir, run_id, log):
    """Submits localmaxxing speed runs on its two canonical prompts, validating each payload before it goes out."""
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
            completed = subprocess.run(command, capture_output=True, text=True)
            window = sampler.window(started, time.time())
            if completed.returncode != 0 or not out.exists():
                log(f"localmaxxing: speed run {prompt_id} failed: {(completed.stdout + completed.stderr)[-400:]}")
                continue
            payload = json.loads(out.read_text())
            flags = payload.setdefault("engineFlags", {})
            flags["commandSnippet"] = profile.cmd[:3900]
            flags.update(spec_counts(profile, payload.get("outputTokens") or 512,
                                     swap.accept_length(before, swap.spec_counters(profile.id)) if before else None))
            meta = profile.engine_meta or {}
            payload.update({key: value for key, value in (("engineRepository", meta.get("repository")),
                                                           ("engineCommit", meta.get("commit")),
                                                           ("engineVersion", meta.get("version"))) if value})
            if window.get("vram_mib"):
                payload["peakVramGb"] = round(window["vram_mib"] / 1024, 1)
            if window.get("power_w"):
                payload["gpuPowerWatts"] = [window["power_w"]]
            payload["notes"] = f"mbench {__version__}, run {run_id}, single request, canonical prompt."
            out.write_text(json.dumps(payload, indent=2))
            check = subprocess.run([LMX, "speed-test", "dry-run", str(out)], capture_output=True, text=True)
            if check.returncode != 0:
                log(f"localmaxxing: speed run {prompt_id} rejected by dry-run: {(check.stdout + check.stderr)[-400:]}")
                continue
            sent = subprocess.run([LMX, "speed-test", "submit", str(out)], capture_output=True, text=True)
            match = re.search(r"\bid: (\w+)", sent.stdout + sent.stderr)
            submitted.append({"kind": f"speed.{prompt_id}", "remote_id": match.group(1) if match else None,
                              "value": payload.get("tokSOut"), "ok": sent.returncode == 0})
            log(f"localmaxxing: speed {prompt_id} {payload.get('tokSOut')} tok/s submitted ({'ok' if sent.returncode == 0 else 'failed'})")
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
    """GSM8K straight to llama-swap; HellaSwag through the log-prob shim. --all-missing keeps already-covered shards from resubmitting."""
    folder = run_dir / "lmx"
    folder.mkdir(parents=True, exist_ok=True)
    server, port = start_shim(paths.SWAP_URL)
    results = {}
    try:
        for dataset, base in (("gsm8k", paths.SWAP_URL), ("hellaswag", f"http://127.0.0.1:{port}")):
            command = [
                LMX, "eval", "shard", dataset, "--base-url", base, "--model", profile.hf_id, "--served-model", profile.id,
                "--hardware", str(hardware_file()), "--quantization", profile.quantization, "--thinking-level", effort,
                "--max-tokens", "16384", "--concurrency", str(suite.CONCURRENCY), "--all-missing", "--submit",
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            text = completed.stdout + completed.stderr
            (folder / f"shards-{dataset}.log").write_text(text)
            results[dataset] = parse_shards(text)
            log(f"localmaxxing: {dataset} submitted {len(results[dataset])} shards")
    finally:
        server.shutdown()
    return results
