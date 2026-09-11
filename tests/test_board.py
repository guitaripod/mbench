from mbench import board


def run(model, suite, effort, stamp):
    return {"id": f"{model}-{suite}-{effort}-{stamp}", "model": model, "suite": suite, "effort": effort, "created": stamp}


def test_each_effort_is_ranked_on_its_own():
    runs = [run("a", "full/v1", "high", 3), run("a", "full/v1", "medium", 2), run("a", "legacy/v0", "medium", 1)]
    assert board.headline(runs, "medium")["a"]["id"] == "a-full/v1-medium-2"
    assert board.headline(runs, "high")["a"]["id"] == "a-full/v1-high-3"
    assert board.headline(runs, "low") == {}


def test_an_older_full_run_beats_a_newer_quick_one():
    runs = [run("a", "quick/v1", "medium", 5), run("a", "full/v1", "medium", 4)]
    assert board.headline(runs, "medium")["a"]["suite"] == "full/v1"


def test_smoke_runs_never_rank_and_missing_effort_means_medium():
    runs = [run("a", "smoke/v1", "medium", 9), {**run("b", "legacy/v0", None, 1), "effort": None}]
    chosen = board.headline(runs, "medium")
    assert "a" not in chosen
    assert chosen["b"]["suite"] == "legacy/v0"
