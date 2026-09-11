from mbench import metrics, suite


def record(item_id, score, tokens=100, finish="stop"):
    return {"id": item_id, "score": score, "completion_tokens": tokens, "latency": 1.0, "finish": finish}


def test_wilson_interval_brackets_the_rate():
    lo, hi = metrics.wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88


def test_repeated_samples_are_grouped_per_problem():
    records = [record(f"aime-{problem}-s{sample}", 1.0 if sample == 0 else 0.0) for problem in range(3) for sample in range(2)]
    score = metrics.task_metrics("aime", records)["aime.score"]
    assert score["n"] == 3
    assert abs(score["value"] - 50.0) < 1e-9


def test_lcb_scores_come_from_grading():
    records = [record("lcb-1", None), record("lcb-2", None, finish="length")]
    out = metrics.task_metrics("lcb", records, {"lcb-1": True, "lcb-2": False})
    assert out["lcb.score"]["value"] == 50.0
    assert out["lcb.truncated"]["value"] == 50.0


def test_index_waits_for_every_task():
    assert metrics.quality_index({"mmlupro.score": {"value": 80.0, "lo": 75.0, "hi": 85.0}}) == {}
    full = {f"{task}.score": {"value": 80.0, "lo": 70.0, "hi": 90.0} for task in suite.INDEX_TASKS}
    index = metrics.quality_index(full)["index.quality"]
    assert (index["value"], index["lo"], index["hi"]) == (80.0, 70.0, 90.0)


def test_speed_metrics_energy_and_depth():
    result = {
        "single": [
            {"prompt": "code-v1", "decode_tps": 200.0, "ttft_s": 0.08, "power_w": 340.0},
            {"prompt": "code-v1", "decode_tps": 220.0, "ttft_s": 0.07, "power_w_mean": 340.0},
            {"prompt": "prose-v1", "decode_tps": 180.0, "ttft_s": 0.06},
        ],
        "concurrency": [{"concurrency": 4, "aggregate_tps": 400.0}],
        "depth": [{"depth": 32000, "ttft_s": 1.7, "prefill_tps": 18800.0, "decode_tps": 155.0}],
    }
    out = metrics.speed_metrics(result)
    assert out["speed.decode.code-v1"]["value"] == 210.0
    assert abs(out["speed.energy"]["value"] - 340.0 / 210.0) < 1e-9
    assert out["speed.decode"]["value"] == 195.0
    assert out["speed.ttft.32000"]["value"] == 1.7
    assert out["speed.conc.4"]["value"] == 400.0
