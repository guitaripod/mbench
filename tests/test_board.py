from mbench import board, paths, store, suite


def run(model, suite_label, effort, stamp):
    return {"id": f"{model}-{suite_label}-{effort}-{stamp}", "model": model, "suite": suite_label, "effort": effort,
            "created": stamp}


def test_each_effort_is_ranked_on_its_own():
    runs = [run("a", "full/v2", "high", 3), run("a", "full/v2", "medium", 2), run("a", "quick/v2", "medium", 1)]
    assert board.headline(runs, "medium")["a"]["id"] == "a-full/v2-medium-2"
    assert board.headline(runs, "high")["a"]["id"] == "a-full/v2-high-3"
    assert board.headline(runs, "low") == {}


def test_an_older_full_run_beats_a_newer_quick_one():
    runs = [run("a", "quick/v2", "medium", 5), run("a", "full/v2", "medium", 4)]
    assert board.headline(runs, "medium")["a"]["suite"] == "full/v2"


def test_smoke_runs_never_rank_and_missing_effort_means_medium():
    runs = [run("a", "smoke/v2", "medium", 9), {**run("b", "legacy/v0", None, 1), "effort": None}]
    chosen = board.headline(runs, "medium")
    assert "a" not in chosen
    assert chosen["b"]["suite"] == "legacy/v0"


def test_suite_versions_are_ranked_apart(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(paths, "DB", tmp_path / "bench.db")
    db = store.connect()
    for model, label, created in (("a", "full/v1", 1.0), ("a", "full/v2", 2.0), ("b", "legacy/v0", 3.0)):
        store.insert_run(db, {"id": f"{model}-{label}", "model": model, "suite": label, "effort": "medium",
                              "status": "complete", "created": created, "finished": created})
    store.set_metrics(db, "a-full/v2", {"supergpqa.score": {"value": 50.0}, "mrcr.bin.16384": {"value": 90.0}})
    data = board.collect(db)
    assert set(data["suites"]) == {"1", suite.VERSION}
    assert [model["id"] for model in data["suites"][suite.VERSION]["rankings"]["medium"]] == ["a"]
    assert sorted(model["id"] for model in data["suites"]["1"]["rankings"]["medium"]) == ["a", "b"]
    assert data["suites"]["1"]["indexTasks"] == list(suite.DEFINITIONS["1"]["index_tasks"])
    entry = data["suites"][suite.VERSION]["rankings"]["medium"][0]["tasks"]["supergpqa"]
    assert entry["value"] == 50.0
    assert data["suites"][suite.VERSION]["rankings"]["medium"][0]["tasks"].get("mrcr") is None
    assert data["rankings"] == data["suites"][suite.VERSION]["rankings"]


def test_task_entries_carry_their_breakdown():
    metrics = {"mrcr.score": {"value": 60.0}, "mrcr.bin.16384": {"value": 90.0}, "mrcr.bin.131072": {"value": 20.0},
               "mrcr.unreachable": {"value": 5.0}}
    entry = board.task_entry(metrics, "mrcr")
    assert entry["parts"] == {"16384": 90.0, "131072": 20.0}
    assert entry["unreachable"] == 5.0
