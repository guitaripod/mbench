import json
import re
import statistics

from . import gpu, lmx, metrics, paths, profiles, store

RESULTS = paths.LEGACY_BENCH / "results"
MODELS = {"sglang-gptoss120b": ("gptoss", "gptoss-dflash6"), "sglang-27b": ("qwen", "qwen-dflash2")}
SHARD_LOGS = {
    "gptoss": {"gsm8k": "lmx-shard-gsm8k-gptoss.log", "hellaswag": "lmx-shard-hellaswag-gptoss.log"},
    "qwen": {"gsm8k": "lmx-shard-gsm8k-qwen.log", "hellaswag": "lmx-shard-hellaswag-qwen-shim.log"},
}
NOTES = "Imported from the 11 Sep 2026 comparison: AIME with 2 samples and LiveCodeBench at a 40k-token budget, no per-item seeds."


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def convert(task, rows, graded=None):
    records = []
    for row in rows:
        match = re.search(r"-s(\d+)$", row["id"])
        score = (1.0 if (graded or {}).get(row["id"]) else 0.0) if task == "lcb" else float(row["correct"])
        records.append({
            "id": row["id"], "task": task, "sample": int(match.group(1)) if match else 0, "gold": row.get("gold"),
            "meta": row.get("meta"), "finish": row.get("finish_reason"), "prompt_tokens": row.get("prompt_tokens"),
            "completion_tokens": row.get("completion_tokens"), "latency": row.get("latency_s"),
            "prediction": row.get("prediction"), "score": score,
        })
    return records


def convert_tools(rows):
    return [{
        "id": f"tools-{row['case']}-r{row['repeat']}", "task": "tools", "sample": row["repeat"], "gold": row["expected"],
        "meta": {"case": row["case"]}, "finish": None, "prompt_tokens": None, "completion_tokens": None,
        "latency": row.get("latency_s"), "prediction": row["calls"], "score": float(row["correct"]),
    } for row in rows]


def import_all(db):
    """Brings the one-off gpt-oss vs Qwen results in as provisional legacy runs; a full mbench run replaces them on the board."""
    imported = []
    for model_id, (key, lmx_label) in MODELS.items():
        run_id = f"20260911-legacy-{model_id}"
        speed_path = RESULTS / f"speed-{key}-prod.json"
        if store.get_run(db, run_id) or not speed_path.exists():
            continue
        profile = profiles.resolve(model_id)
        run_dir = paths.RUNS / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        collected = {}
        speed = json.loads(speed_path.read_text())
        (run_dir / "speed.json").write_text(json.dumps(speed))
        collected.update(metrics.speed_metrics(speed))
        for task in ("niah", "mmlupro", "aime", "lcb"):
            source = RESULTS / f"{task}-{key}-medium.jsonl"
            if not source.exists():
                continue
            graded = None
            if task == "lcb":
                graded = {row["id"]: row["passed"] for row in read_jsonl(RESULTS / f"lcb-{key}-medium.graded.jsonl")}
            records = convert(task, read_jsonl(source), graded)
            (run_dir / f"{task}.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
            collected.update(metrics.task_metrics(task, records, graded))
        tools_path = RESULTS / f"toolcall-{key}-medium.json"
        if tools_path.exists():
            records = convert_tools(json.loads(tools_path.read_text()))
            (run_dir / "tools.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
            collected.update(metrics.task_metrics("tools", records))
        collected.update(metrics.quality_index(collected))
        finished = speed_path.stat().st_mtime
        store.insert_run(db, {
            "id": run_id, "model": model_id, "name": profile.name, "suite": "legacy/v0", "effort": "medium",
            "status": "complete", "created": finished, "started": finished, "finished": finished,
            "harness": "bench-2026-09-11", "fingerprint": profile.fingerprint, "profile": profile.to_dict(),
            "hardware": gpu.describe(), "flags": {"legacy": True, "notes": NOTES},
        })
        for dataset, name in SHARD_LOGS[key].items():
            path = RESULTS / name
            entries = lmx.parse_shards(path.read_text()) if path.exists() else []
            for entry in entries:
                store.add_submission(db, run_id, f"shard.{dataset}", str(entry["shard"]), entry["accuracy"], entry)
            if entries:
                collected[f"lmx.{dataset}"] = {"value": statistics.mean(entry["accuracy"] for entry in entries),
                                               "unit": "%", "n": len(entries)}
        for prompt in ("code-v1", "reasoning-v1"):
            path = RESULTS / f"lmx-{lmx_label}-{prompt}.json"
            if path.exists():
                payload = json.loads(path.read_text())
                store.add_submission(db, run_id, f"speed.{prompt}", payload.get("submissionId"), payload.get("tokSOut"))
        store.set_metrics(db, run_id, collected)
        imported.append(run_id)
    return imported

