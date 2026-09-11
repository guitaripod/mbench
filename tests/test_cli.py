import json

import pytest

from mbench import cli, paths


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
