import json
import math
import re
import statistics
from collections import defaultdict

from . import suite

BINARY_TASKS = ("mmlupro", "lcb", "aime", "tools")


def wilson(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - margin, centre + margin


def mean_interval(values, z=1.96):
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean, mean
    error = statistics.stdev(values) / math.sqrt(len(values))
    return mean, max(0.0, mean - z * error), min(1.0, mean + z * error)


def load(run_dir, task):
    path = run_dir / f"{task}.jsonl"
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def base_id(item_id):
    return re.sub(r"-(s|r)\d+$", "", item_id)


def task_metrics(task, records, graded=None):
    """Score with a 95% interval: Wilson for single-sample pass/fail tasks, otherwise the spread of per-question means."""
    if not records:
        return {}
    if task == "lcb":
        scores = {record["id"]: 1.0 if (graded or {}).get(record["id"]) else 0.0 for record in records}
    else:
        scores = {record["id"]: float(record["score"] or 0.0) for record in records}
    groups = defaultdict(list)
    for item_id, score in scores.items():
        groups[base_id(item_id)].append(score)
    per_question = [statistics.mean(values) for values in groups.values()]
    single_sample = all(len(values) == 1 for values in groups.values())
    if task in BINARY_TASKS and single_sample:
        value = statistics.mean(per_question)
        lo, hi = wilson(sum(per_question), len(per_question))
    else:
        value, lo, hi = mean_interval(per_question)
    tokens = [record["completion_tokens"] for record in records if record.get("completion_tokens") is not None]
    latencies = [record["latency"] for record in records if record.get("latency") is not None]
    out = {f"{task}.score": {"value": 100 * value, "unit": "%", "n": len(per_question), "lo": 100 * lo, "hi": 100 * hi}}
    if tokens:
        out[f"{task}.tokens"] = {"value": statistics.mean(tokens), "unit": "tokens", "n": len(tokens)}
    if latencies:
        out[f"{task}.latency"] = {"value": statistics.median(latencies), "unit": "s", "n": len(latencies)}
    truncated = [record.get("finish") == "length" for record in records]
    out[f"{task}.truncated"] = {"value": 100 * sum(truncated) / len(truncated), "unit": "%", "n": len(truncated)}
    return out


def quality_index(existing):
    """Mean of the five index tasks; left out until every one of them has a score, so partial runs never rank."""
    entries = [existing.get(f"{task}.score") for task in suite.INDEX_TASKS]
    if any(entry is None or entry.get("value") is None for entry in entries):
        return {}
    return {"index.quality": {
        "value": statistics.mean(entry["value"] for entry in entries),
        "unit": "%",
        "n": len(entries),
        "lo": statistics.mean(entry["lo"] if entry.get("lo") is not None else entry["value"] for entry in entries),
        "hi": statistics.mean(entry["hi"] if entry.get("hi") is not None else entry["value"] for entry in entries),
    }}


def power_of(row):
    return row.get("power_w", row.get("power_w_mean"))


def spread(values, unit):
    return {"value": statistics.median(values), "unit": unit, "n": len(values), "lo": min(values), "hi": max(values)}


def speed_metrics(result):
    out = {}
    single = defaultdict(list)
    for row in result.get("single", []):
        if row.get("decode_tps"):
            single[row["prompt"]].append(row)
    medians = []
    for prompt, rows in single.items():
        entry = spread([row["decode_tps"] for row in rows], "tok/s")
        out[f"speed.decode.{prompt}"] = entry
        medians.append(entry["value"])
        accepts = [row["accept_length"] for row in rows if row.get("accept_length")]
        if accepts:
            out[f"speed.accept.{prompt}"] = spread(accepts, "tokens/step")
    if medians:
        out["speed.decode"] = {"value": statistics.mean(medians), "unit": "tok/s", "n": len(medians)}
    ttfts = [row["ttft_s"] for rows in single.values() for row in rows if row.get("ttft_s")]
    if ttfts:
        out["speed.ttft.short"] = spread([1000 * value for value in ttfts], "ms")
    code_rows = single.get("code-v1", [])
    powers = [power_of(row) for row in code_rows if power_of(row)]
    if powers and "speed.decode.code-v1" in out:
        power = statistics.median(powers)
        out["speed.power"] = {"value": power, "unit": "W", "n": len(powers)}
        out["speed.energy"] = {"value": power / out["speed.decode.code-v1"]["value"], "unit": "J/token", "n": len(powers)}
    concurrency = defaultdict(list)
    for row in result.get("concurrency", []):
        if row.get("aggregate_tps"):
            concurrency[row["concurrency"]].append(row["aggregate_tps"])
    for level, values in concurrency.items():
        out[f"speed.conc.{level}"] = spread(values, "tok/s")
    depth = defaultdict(list)
    for row in result.get("depth", []):
        if row.get("ttft_s"):
            depth[row["depth"]].append(row)
    for level, rows in depth.items():
        out[f"speed.ttft.{level}"] = spread([row["ttft_s"] for row in rows], "s")
        prefill = [row["prefill_tps"] for row in rows if row.get("prefill_tps")]
        if prefill:
            out[f"speed.prefill.{level}"] = spread(prefill, "tok/s")
        decode = [row["decode_tps"] for row in rows if row.get("decode_tps")]
        if decode:
            out[f"speed.decode_at.{level}"] = spread(decode, "tok/s")
    return out
