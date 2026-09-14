import json
import math
import re
import statistics
from collections import defaultdict

import numpy as np

from . import suite

BINARY_TASKS = ("supergpqa", "math", "lcb", "tools")
THERMAL_STATES = ("nominal", "fair", "serious", "critical")
HOLDS_RATIO = 0.8
BARELY_RATIO = 0.55
BARELY_TOKENS_PER_SECOND = 8.0
HOLDS_TOKENS_TO_KNEE = 2048
BARELY_TOKENS_TO_KNEE = 512
THROTTLE_SHARE = 0.9
BOOTSTRAP_DRAWS = 4000


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


def graded_of(run_dir):
    path = run_dir / "lcb.graded.jsonl"
    if not path.exists():
        return {}
    with path.open() as handle:
        return {row["id"]: row["passed"] for row in map(json.loads, handle) if row}


def base_id(item_id):
    return re.sub(r"-(s|r)\d+$", "", item_id)


def item_scores(task, records, graded=None):
    if task == "lcb":
        return {record["id"]: 1.0 if (graded or {}).get(record["id"]) else 0.0 for record in records}
    return {record["id"]: float(record["score"] or 0.0) for record in records}


def question_scores(task, records, graded=None):
    """Each question's mean over its samples or repeats, so a question asked four times still counts once."""
    groups = defaultdict(list)
    for item_id, score in item_scores(task, records, graded).items():
        groups[base_id(item_id)].append(score)
    return {question: statistics.mean(values) for question, values in groups.items()}


def run_question_scores(run_dir, task):
    return question_scores(task, load(run_dir, task), graded_of(run_dir) if task == "lcb" else None)


def percent(values, unit="%"):
    return {"value": 100 * statistics.mean(values), "unit": unit, "n": len(values)}


def breakdown(task, records, scores):
    """Scores by length bin for the long-context tasks, by category for tool use and by competition for math."""
    field = {"mrcr": "bin", "graphwalks": "bin", "tools": "category", "math": "competition"}.get(task)
    if not field:
        return {}
    groups = defaultdict(list)
    for record in records:
        groups[record["meta"].get(field)].append(scores[record["id"]])
    return {f"{task}.{'bin' if field == 'bin' else 'part'}.{key}": percent(values)
            for key, values in groups.items() if key is not None}


def task_metrics(task, records, graded=None):
    """Score with a 95% interval: Wilson for single-sample pass/fail tasks, otherwise the spread of per-question means."""
    if not records:
        return {}
    scores = item_scores(task, records, graded)
    per_question = list(question_scores(task, records, graded).values())
    single_sample = len(per_question) == len(scores)
    if task in BINARY_TASKS and single_sample:
        value = statistics.mean(per_question)
        lo, hi = wilson(sum(per_question), len(per_question))
    else:
        value, lo, hi = mean_interval(per_question)
    tokens = [record["completion_tokens"] for record in records if record.get("completion_tokens") is not None]
    latencies = [record["latency"] for record in records if record.get("finish") != "context" and record.get("latency")]
    out = {f"{task}.score": {"value": 100 * value, "unit": "%", "n": len(per_question), "lo": 100 * lo, "hi": 100 * hi}}
    if tokens:
        out[f"{task}.tokens"] = {"value": statistics.mean(tokens), "unit": "tokens", "n": len(tokens)}
    if latencies:
        out[f"{task}.latency"] = {"value": statistics.median(latencies), "unit": "s", "n": len(latencies)}
    out[f"{task}.truncated"] = percent([record.get("finish") == "length" for record in records])
    if any(record.get("finish") == "context" for record in records):
        out[f"{task}.unreachable"] = percent([record.get("finish") == "context" for record in records])
    if task == "tools":
        out["tools.malformed"] = percent([bool(record.get("malformed")) for record in records])
    out.update(breakdown(task, records, scores))
    return out


def resample_means(values, rng, draws=BOOTSTRAP_DRAWS):
    values = np.asarray(values, dtype=float)
    return values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)


def combine(task_means, groups):
    """The quality index from per-task means: tasks average within their group, groups average equally."""
    return np.mean([np.mean([task_means[task] for task in tasks], axis=0) for tasks in groups.values()], axis=0)


def quality_index(existing, run_dir, groups=None):
    """Mean of the index groups, left out until every task has a score; its interval bootstraps questions within each task."""
    groups = groups or suite.INDEX_GROUPS
    tasks = [task for members in groups.values() for task in members]
    if any((existing.get(f"{task}.score") or {}).get("value") is None for task in tasks):
        return {}
    per_task = {task: list(run_question_scores(run_dir, task).values()) for task in tasks}
    if any(not values for values in per_task.values()):
        return {}
    rng = np.random.default_rng(0)
    draws = combine({task: resample_means(values, rng) for task, values in per_task.items()}, groups)
    value = combine({task: np.mean(values) for task, values in per_task.items()}, groups)
    return {"index.quality": {
        "value": 100 * float(value), "unit": "%", "n": len(groups),
        "lo": 100 * float(np.percentile(draws, 2.5)), "hi": 100 * float(np.percentile(draws, 97.5)),
    }}


def paired(scores_a, scores_b, rng=None):
    """A minus B over the questions both runs answered, with a bootstrap 95% interval on the difference."""
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        return None
    a = np.array([scores_a[question] for question in shared])
    b = np.array([scores_b[question] for question in shared])
    draws = resample_means(a - b, rng or np.random.default_rng(0))
    return {"n": len(shared), "a": 100 * a.mean(), "b": 100 * b.mean(), "diff": 100 * (a - b).mean(),
            "lo": 100 * float(np.percentile(draws, 2.5)), "hi": 100 * float(np.percentile(draws, 97.5))}


def paired_index(tasks_a, tasks_b, groups=None):
    """The quality-index difference, resampling the same shared questions for both runs so their correlation cancels."""
    groups = groups or suite.INDEX_GROUPS
    rng = np.random.default_rng(0)
    diffs, means = {}, {}
    for task in (task for members in groups.values() for task in members):
        shared = sorted(set(tasks_a.get(task, {})) & set(tasks_b.get(task, {})))
        if not shared:
            return None
        delta = np.array([tasks_a[task][question] - tasks_b[task][question] for question in shared])
        diffs[task], means[task] = resample_means(delta, rng), delta.mean()
    draws = combine(diffs, groups)
    return {"diff": 100 * float(combine(means, groups)),
            "lo": 100 * float(np.percentile(draws, 2.5)), "hi": 100 * float(np.percentile(draws, 97.5))}


def energy_metrics(run_dir):
    """Board energy over the quality tasks (idle draw included) and per correct answer, partial credit counting in part."""
    path = run_dir / "energy.json"
    if not path.exists():
        return {}
    energy = json.loads(path.read_text())
    graded = graded_of(run_dir)
    correct = sum(sum(item_scores(task, load(run_dir, task), graded if task == "lcb" else None).values()) for task in energy)
    total = sum(energy.values())
    out = {f"{task}.energy": {"value": wh, "unit": "Wh", "n": 1} for task, wh in energy.items()}
    out["energy.quality"] = {"value": total, "unit": "Wh", "n": len(energy)}
    if correct:
        out["energy.per_correct"] = {"value": total / correct, "unit": "Wh", "n": round(correct)}
    return out


def task_health(models, tasks, minimum=3):
    """Tasks that don't separate the ranked models: all near the ceiling, all near the floor, or spread inside the noise."""
    health = {}
    for task in tasks:
        entries = [model["tasks"][task] for model in models if model["tasks"].get(task)]
        if len(entries) < minimum:
            continue
        values = [entry["value"] for entry in entries]
        halves = [(entry["hi"] - entry["lo"]) / 2 for entry in entries
                  if entry.get("lo") is not None and entry.get("hi") is not None]
        if min(values) >= 90:
            health[task] = "near the ceiling for every model"
        elif max(values) <= 10:
            health[task] = "near the floor for every model"
        elif halves and max(values) - min(values) < statistics.mean(halves):
            health[task] = "models differ by less than the noise"
    return health


def power_of(row):
    return row.get("power_w", row.get("power_w_mean"))


def spread(values, unit):
    return {"value": statistics.median(values), "unit": unit, "n": len(values), "lo": min(values), "hi": max(values)}


def sustain_metrics(rows):
    """What back-to-back decoding costs: the cold speed, the speed it settles at, the ratio between them and how long
    the phone held the cold one. On a desktop the two are the same number."""
    paired = [(row["index"], row["decode_tps"], row) for row in rows if row.get("decode_tps")]
    if len(paired) < 4:
        return {}
    out = {}
    for index, value, row in paired:
        out[f"speed.sustain.{index:02d}"] = {"value": value, "unit": "tok/s", "n": 1}
        if row.get("thermal") in THERMAL_STATES:
            out[f"speed.sustain_state.{index:02d}"] = {"value": THERMAL_STATES.index(row["thermal"]), "unit": "state", "n": 1}
    values = [value for _, value, _ in paired]
    peak = max(values[:2])
    tail = values[len(values) * 2 // 3:]
    out["speed.decode_peak"] = {"value": peak, "unit": "tok/s", "n": min(2, len(values))}
    out["speed.decode_steady"] = spread(tail, "tok/s")
    out["speed.sustain_ratio"] = {"value": statistics.median(tail) / peak, "unit": "x", "n": len(values)}
    started = paired[0][2]["start"]
    delivered = 0
    for _, value, row in paired:
        if value < THROTTLE_SHARE * peak:
            out["speed.throttle_s"] = {"value": round(row["start"] - started, 1), "unit": "s", "n": 1}
            out["speed.tokens_to_knee"] = {"value": delivered, "unit": "tokens", "n": 1}
            break
        delivered += row.get("completion_tokens") or 0
    return out


def thermal_metrics(timeline, tokens=None):
    """How a phone behaves under load: how long it stays cool, how much of the run it spends hot, and what the
    answers cost the battery. A desktop has no timeline, so none of this appears for one."""
    samples = [entry for entry in timeline or [] if entry.get("thermal") in THERMAL_STATES]
    if len(samples) < 3:
        return {}
    started = samples[0]["t"]
    worst = max(samples, key=lambda entry: THERMAL_STATES.index(entry["thermal"]))["thermal"]
    hot = [entry for entry in samples if THERMAL_STATES.index(entry["thermal"]) >= 2]
    out = {
        "thermal.worst": {"value": THERMAL_STATES.index(worst), "unit": "state", "n": len(samples)},
        "thermal.share_hot": {"value": 100 * len(hot) / len(samples), "unit": "%", "n": len(samples)},
    }
    for name in ("fair", "serious", "critical"):
        reached = next((entry for entry in samples if THERMAL_STATES.index(entry["thermal"]) >= THERMAL_STATES.index(name)), None)
        if reached:
            out[f"thermal.to_{name}_s"] = {"value": round(reached["t"] - started, 1), "unit": "s", "n": 1}
    out.update(battery_metrics(samples, tokens))
    return out


def battery_metrics(samples, tokens=None):
    """What a run cost the battery. Only a run on battery can say: charging hides the drain entirely."""
    levels = [entry["battery"] for entry in samples if entry.get("battery") is not None]
    on_battery = [entry for entry in samples if entry.get("battery_state") == "unplugged"]
    if len(levels) < 2 or len(on_battery) < len(samples):
        return {}
    drop = 100 * (levels[0] - levels[-1])
    if drop <= 0:
        return {}
    out = {"battery.drain": {"value": drop, "unit": "%", "n": len(levels)}}
    elapsed = samples[-1]["t"] - samples[0]["t"]
    if elapsed > 60:
        out["battery.per_hour"] = {"value": drop * 3600 / elapsed, "unit": "%/h", "n": len(levels)}
    if tokens:
        out["battery.per_1k_tokens"] = {"value": 1000 * drop / tokens, "unit": "%/1k", "n": len(levels)}
    return out


def answered_tokens(result):
    rows = [row for key in ("single", "concurrency", "depth", "sustain") for row in result.get(key, [])]
    streams = [stream for row in rows for stream in (row.get("streams") or [row])]
    return sum(stream.get("completion_tokens") or 0 for stream in streams) or None


def footprint_metrics(result):
    """The most memory the phone's process held while measuring, which is what decides whether a model fits at all."""
    rows = [row for key in ("single", "concurrency", "depth", "sustain") for row in result.get(key, [])]
    peaks = [row["footprint_mib"] for row in rows if row.get("footprint_mib")]
    return {"speed.footprint": {"value": max(peaks), "unit": "MiB", "n": len(peaks)}} if peaks else {}


def speed_metrics(result):
    out = {}
    out.update(sustain_metrics(result.get("sustain", [])))
    out.update(footprint_metrics(result))
    out.update(thermal_metrics(result.get("telemetry"), answered_tokens(result)))
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
    temps = [row["temp_c"] for row in code_rows if row.get("temp_c")]
    if temps:
        out["speed.temp"] = {"value": statistics.median(temps), "unit": "C", "n": len(temps)}
    concurrency = defaultdict(list)
    for row in result.get("concurrency", []):
        if row.get("aggregate_tps"):
            concurrency[row["concurrency"]].append(row["aggregate_tps"])
    for level, values in concurrency.items():
        out[f"speed.conc.{level}"] = spread(values, "tok/s")
    if concurrency:
        best = max(concurrency, key=lambda level: statistics.median(concurrency[level]))
        out["speed.peak"] = spread(concurrency[best], "tok/s")
        out["speed.peak_at"] = {"value": best, "unit": "requests", "n": len(concurrency[best])}
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


def verdict(entry):
    """Whether a phone can live with a model, decided only on tokens the model itself produced: what it settles at,
    how much of its cold speed that is, and how many tokens it delivered before it gave any of it up. The phone's
    own thermal state stays out of it — a charger, a warm room or a different phone move that number, and none of
    them move this one."""
    speed = {key.removeprefix("speed."): value for key, value in entry.items() if key.startswith("speed.")}
    steady = (speed.get("decode_steady") or {}).get("value")
    ratio = (speed.get("sustain_ratio") or {}).get("value")
    knee = (speed.get("tokens_to_knee") or {}).get("value")
    if steady is None and ratio is None:
        return None
    if ((steady is not None and steady < BARELY_TOKENS_PER_SECOND)
            or (ratio is not None and ratio < BARELY_RATIO)
            or (knee is not None and knee < BARELY_TOKENS_TO_KNEE)):
        return "barely"
    if ((ratio is not None and ratio < HOLDS_RATIO)
            or (knee is not None and knee < HOLDS_TOKENS_TO_KNEE)):
        return "fades"
    return "holds"
