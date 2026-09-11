import json

from mbench import metrics, suite


def record(item_id, score, tokens=100, finish="stop", meta=None):
    return {"id": item_id, "score": score, "completion_tokens": tokens, "latency": 1.0, "finish": finish, "meta": meta or {}}


def write(run_dir, task, records):
    (run_dir / f"{task}.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in records))


def test_wilson_interval_brackets_the_rate():
    lo, hi = metrics.wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88


def test_repeated_samples_are_grouped_per_problem():
    records = [record(f"aime26-{problem}-s{sample}", 1.0 if sample == 0 else 0.0) for problem in range(3) for sample in range(2)]
    score = metrics.task_metrics("math", records)["math.score"]
    assert score["n"] == 3
    assert abs(score["value"] - 50.0) < 1e-9


def test_lcb_scores_come_from_grading():
    records = [record("lcb-1", None), record("lcb-2", None, finish="length")]
    out = metrics.task_metrics("lcb", records, {"lcb-1": True, "lcb-2": False})
    assert out["lcb.score"]["value"] == 50.0
    assert out["lcb.truncated"]["value"] == 50.0


def test_long_context_scores_break_down_by_length_and_count_unreachable():
    records = [record("mrcr-16384-a", 1.0, meta={"bin": 16384}), record("mrcr-16384-b", 0.5, meta={"bin": 16384}),
               record("mrcr-131072-c", 0.0, tokens=None, finish="context", meta={"bin": 131072})]
    out = metrics.task_metrics("mrcr", records)
    assert out["mrcr.bin.16384"]["value"] == 75.0
    assert out["mrcr.bin.131072"]["value"] == 0.0
    assert abs(out["mrcr.unreachable"]["value"] - 100 / 3) < 1e-9
    assert abs(out["mrcr.score"]["value"] - 50.0) < 1e-9


def test_tool_scores_break_down_by_category():
    records = [record(f"tools-a-r{repeat}", 1.0, meta={"category": "policy"}) for repeat in range(3)]
    records += [{**record("tools-b-r0", 0.0, meta={"category": "clarify"}), "malformed": 1}]
    out = metrics.task_metrics("tools", records)
    assert out["tools.part.policy"]["value"] == 100.0
    assert out["tools.part.clarify"]["value"] == 0.0
    assert out["tools.malformed"]["value"] == 25.0
    assert out["tools.score"]["n"] == 2


def test_index_waits_for_every_task(tmp_path):
    assert metrics.quality_index({"supergpqa.score": {"value": 80.0}}, tmp_path) == {}


def full_run(run_dir, score_of):
    existing = {}
    for task in suite.INDEX_TASKS:
        records = [record(f"{task}-{index}", score_of(task, index)) for index in range(40)]
        write(run_dir, task, [{**entry, "score": None} for entry in records] if task == "lcb" else records)
        if task == "lcb":
            (run_dir / "lcb.graded.jsonl").write_text("".join(
                json.dumps({"id": entry["id"], "passed": entry["score"] == 1.0}) + "\n" for entry in records))
        existing[f"{task}.score"] = {"value": 1.0}
    return existing


def test_index_averages_groups_and_bootstraps_an_interval(tmp_path):
    existing = full_run(tmp_path, lambda task, index: 1.0 if index % 2 == 0 or task == "mrcr" else 0.0)
    index = metrics.quality_index(existing, tmp_path)["index.quality"]
    assert abs(index["value"] - 55.0) < 1e-9
    assert index["lo"] < index["value"] < index["hi"]
    assert index["hi"] - index["lo"] < 20


def test_a_perfect_run_has_no_spread(tmp_path):
    index = metrics.quality_index(full_run(tmp_path, lambda task, index: 1.0), tmp_path)["index.quality"]
    assert (index["value"], index["lo"], index["hi"]) == (100.0, 100.0, 100.0)


def test_paired_difference_and_its_interval():
    a = {f"q{index}": 1.0 for index in range(50)}
    b = {f"q{index}": 1.0 if index < 25 else 0.0 for index in range(50)}
    better = metrics.paired(a, b)
    assert better["diff"] == 50.0 and better["lo"] > 0
    same = metrics.paired(a, dict(a))
    assert same["diff"] == 0.0 and same["lo"] <= 0 <= same["hi"]
    assert metrics.paired(a, {"other": 1.0}) is None


def test_paired_index_needs_shared_questions_in_every_task():
    tasks = {task: {"q1": 1.0, "q2": 0.0} for task in suite.INDEX_TASKS}
    assert metrics.paired_index(tasks, tasks)["diff"] == 0.0
    assert metrics.paired_index(tasks, {**tasks, "tools": {}}) is None


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
