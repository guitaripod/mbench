import json

import pytest

from mbench import cli, paths, suite
from mbench.profiles import Profile


def test_a_missing_llama_swap_config_is_a_clear_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(paths, "LLAMA_SWAP_CONFIG", tmp_path / "config.yaml")
    monkeypatch.setattr("sys.argv", ["mbench", "profile", "some-model"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    assert "MBENCH_SWAP_CONFIG" in capsys.readouterr().err


def test_a_git_install_records_the_commit_it_was_built_from(monkeypatch):
    class Distribution:
        def read_text(self, name):
            return json.dumps({"url": "https://github.com/guitaripod/mbench",
                               "vcs_info": {"vcs": "git", "commit_id": "5fbf2e1c0ffee0ddba11"}})

    monkeypatch.setattr(cli.metadata, "distribution", lambda name: Distribution())
    assert cli.installed_commit() == "5fbf2e1"


def test_a_directory_outside_any_checkout_has_no_checkout_commit(tmp_path):
    assert cli.checkout_commit(tmp_path) is None


def test_only_unchanged_tasks_carry_over_between_suite_versions():
    assert suite.reusable("speed", "1") and suite.reusable("lcb", "1")
    assert not suite.reusable("tools", "1") and not suite.reusable("supergpqa", "1")
    assert all(suite.reusable(task, suite.VERSION) for task in ("speed", *suite.QUALITY_TASKS))
    assert suite.version_of("legacy/v0") == "1" and suite.version_of("full/v2") == "2"


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
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    (tmp_path / "r1").mkdir()
    (tmp_path / "r2").mkdir()
    for name in ("speed.json", "lcb.jsonl", "lcb.graded.jsonl", "tools.jsonl", "aime.jsonl"):
        (tmp_path / "r1" / name).write_text("{}\n")
    carried = cli.carry_over(earlier(), "r2", ["speed", "supergpqa", "tools", "lcb"])
    assert carried == ["speed", "lcb"]
    assert sorted(path.name for path in (tmp_path / "r2").iterdir()) == ["lcb.graded.jsonl", "lcb.jsonl", "speed.json"]
