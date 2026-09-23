import json
from types import SimpleNamespace

import pytest

from mbench import cli, paths, store, suite
from mbench.profiles import Profile


def test_a_missing_llama_swap_config_is_a_clear_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(paths, "LLAMA_SWAP_CONFIG", tmp_path / "config.yaml")
    monkeypatch.setattr("sys.argv", ["mbench", "profile", "some-model"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    assert "MBENCH_SWAP_CONFIG" in capsys.readouterr().err


OLDER = {"0": {"task_versions": {"speed": 1, "lcb": 1, "tools": 0}}}


def test_only_unchanged_tasks_carry_over_between_suite_versions(monkeypatch):
    monkeypatch.setattr(suite, "DEFINITIONS", {**suite.DEFINITIONS, **OLDER})
    assert suite.reusable("speed", "0") and suite.reusable("lcb", "0")
    assert not suite.reusable("tools", "0") and not suite.reusable("supergpqa", "0")
    assert all(suite.reusable(task, suite.VERSION) for task in ("speed", *suite.QUALITY_TASKS))
    assert not suite.reusable("speed", "9")
    assert suite.version_of("full/v1") == "1"


def profile():
    return Profile(id="m", name="m", engine="sglang", thinking="qwen", context=None, efforts=["medium"],
                   fingerprint="abc")


def earlier(**overrides):
    return {"id": "r1", "model": "m", "status": "failed", "suite": "full/v1", "effort": "medium",
            "flags": {"effort_level": "medium"}, "fingerprint": "abc", **overrides}


def test_reuse_needs_the_same_model_config_effort_and_suite_size():
    assert cli.reuse_problem(earlier(), profile(), "medium", "full") is None
    assert "measured" in cli.reuse_problem(earlier(model="x"), profile(), "medium", "full")
    assert "quick" in cli.reuse_problem(earlier(suite="quick/v1"), profile(), "medium", "full")
    assert "effort" in cli.reuse_problem(earlier(flags={"effort_level": "xhigh"}), profile(), "medium", "full")
    assert "changed" in cli.reuse_problem(earlier(fingerprint="zzz"), profile(), "medium", "full")
    assert "running" in cli.reuse_problem(earlier(status="running"), profile(), "medium", "full")


def test_carry_over_copies_only_files_that_still_apply(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "DEFINITIONS", {**suite.DEFINITIONS, **OLDER})
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    (tmp_path / "r1").mkdir()
    (tmp_path / "r2").mkdir()
    for name in ("speed.json", "lcb.jsonl", "lcb.graded.jsonl", "tools.jsonl", "aime.jsonl"):
        (tmp_path / "r1" / name).write_text("{}\n")
    carried = cli.carry_over(earlier(suite="full/v0"), "r2", ["speed", "supergpqa", "tools", "lcb"])
    assert carried == ["speed", "lcb"]
    assert sorted(path.name for path in (tmp_path / "r2").iterdir()) == ["lcb.graded.jsonl", "lcb.jsonl", "speed.json"]


def test_the_markdown_table_is_a_github_table(capsys):
    view = {"indexTasks": ["math"], "taskShort": {"math": "Math"}, "rankings": {"max": [{
        "name": "A Model", "index": {"value": 81.0}, "tasks": {"math": {"value": 76.25}},
        "speed": {"decode": {"value": 157.4}, "peak": {"value": 400.2}, "ttft.32000": {"value": 5.04}},
        "energy": {"perCorrect": {"value": 2.671}}, "run": {"finished": 1789200000.0, "kind": "full"}}]}}
    header, rows = cli.ranking_table(view, "max")
    cli.print_markdown(header, rows)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("| Model | Quality | Math |")
    assert lines[1] == "|---|--:|--:|--:|--:|--:|--:|--:|"
    assert lines[2] == "| A Model | 81.0 | 76.2 | 157 | 400 | 5.0 | 2.67 | 12 Sep |"


def test_a_ranking_measured_on_two_cards_says_the_speed_columns_dont_compare():
    note = cli.setups_note([{"label": "RTX PRO 6000 · driver 610", "models": 2}, {"label": "RTX 4090 · driver 570", "models": 1}])
    assert note.startswith("tok/s, Peak and Wh/correct come from 2 setups")
    assert "RTX PRO 6000 · driver 610 (2), RTX 4090 · driver 570 (1)" in note


def test_time_left_counts_the_time_spent_answering_and_never_the_time_parked():
    events = [{"t": 0, "phase": "started"}, {"t": 10, "phase": "supergpqa", "done": 0, "total": 100},
              {"t": 1000, "phase": "yielded"}, {"t": 5000, "phase": "started"},
              {"t": 5010, "phase": "supergpqa", "done": 40, "total": 100},
              {"t": 5610, "phase": "supergpqa", "done": 50, "total": 100}]
    assert cli.time_left(events, 6010) == 1990
    assert cli.time_left(events[:5], 5020) == 1500
    assert cli.time_left(events, 90000, running=False) == 1590
    assert cli.time_left(events[:2], 20) is None
    assert cli.time_left([], 10) is None
    assert cli.tasks_after({"flags": {"tasks": ["speed", "supergpqa", "math", "lcb"]}}, "supergpqa") == ["math", "lcb"]
    assert cli.tasks_after({"flags": {}}, "lcb") == []


def test_wait_rides_out_a_run_that_gives_way_and_exits_with_its_verdict(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(paths, "DB", tmp_path / "bench.db")
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    (tmp_path / "runs" / "r1" / "events.jsonl").write_text(
        json.dumps({"t": 1.0, "phase": "math", "done": 3, "total": 126}) + "\n")
    db = store.connect()
    store.insert_run(db, {"id": "r1", "model": "m", "suite": "full/v1", "status": "running", "started": 1.0, "flags": {}})
    later = iter([{"status": "scheduled", "flags": {"not_before": 4e9, "parked": "gave the GPU to Xwayland"}},
                  {"status": "running"}, {"status": "complete"}])
    monkeypatch.setattr(cli, "reconcile", lambda db: None)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: store.update_run(db, "r1", **next(later)))
    monkeypatch.setattr(cli, "summary", lambda db, run_id: print("summary"))
    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_wait(SimpleNamespace(run=None, every=5))
    lines = capsys.readouterr().out.splitlines()
    assert exit_info.value.code == 0
    assert lines[0].startswith("r1  full/v1  math 3/126  running for")
    assert lines[1] == lines[3] == lines[5] == "  then tools, mrcr, graphwalks, lcb"
    assert "gave the GPU to Xwayland; continues where it stopped" in lines[2]
    assert lines[4].startswith("r1  full/v1  math 3/126") and lines[6:] == ["r1: complete", "summary"]
    store.update_run(db, "r1", status="failed", error="the GPU stopped answering")
    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_wait(SimpleNamespace(run="r1", every=5))
    assert exit_info.value.code == 1 and "r1: failed (the GPU stopped answering)" in capsys.readouterr().out
