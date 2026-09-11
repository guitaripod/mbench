from types import SimpleNamespace

from mbench import datasets, notify, sources


def test_the_notify_command_gets_the_text(monkeypatch):
    calls = []
    monkeypatch.setattr(notify.shutil, "which", lambda name: None)
    monkeypatch.setattr(notify.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs.get("env") or {})))
    monkeypatch.setenv("MBENCH_NOTIFY", "push-it")
    notify.send("mbench finished", "gpt-oss: quality index 71.2", run="r1")
    command, environment = calls[0]
    assert command == "push-it"
    assert environment["MBENCH_MESSAGE"] == "gpt-oss: quality index 71.2" and environment["MBENCH_RUN"] == "r1"


def test_the_config_file_supplies_the_command(monkeypatch, tmp_path):
    monkeypatch.delenv("MBENCH_NOTIFY", raising=False)
    monkeypatch.setattr(notify.paths, "SETTINGS", tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text('notify = "curl -d \\"$MBENCH_MESSAGE\\" https://ntfy.example/bench"\n')
    assert notify.hook().startswith("curl")


class Hub:
    def __init__(self, moved=()):
        self.moved = set(moved)

    def get_paths_info(self, repo, files, repo_type):
        pinned = next(source["sha256"] for source in datasets.SOURCES.values()
                      if source["repo"] == repo and source["file"] == files[0])
        return [SimpleNamespace(lfs=SimpleNamespace(sha256="0" * 64 if repo in self.moved else pinned))]

    def list_repo_files(self, repo, repo_type):
        return ["README.md", "test5.jsonl", "test6.jsonl", "test7.jsonl"]

    def list_datasets(self, author, limit):
        names = ("aime_2026", "aime_2026_I", "hmmt_feb_2026", "smt_2025", "arxivmath-0626", "usamo_2026", "aime_2027",
                 "aime_2027_outputs")
        return [SimpleNamespace(id=f"{author}/{name}") for name in names]


def test_sources_report_moved_pins_and_newer_sets():
    hub = Hub(moved={"openai/graphwalks"})
    assert sources.changed_pins(hub) == ["graphwalks: openai/graphwalks/graphwalks_128k_and_shorter.parquet changed "
                                         "upstream since it was pinned"]
    assert sources.newer_lcb(hub) == ["LiveCodeBench: livecodebench/code_generation_lite has test7.jsonl, newer than the "
                                      "pinned test6.jsonl"]
    assert sources.newer_math(hub) == [
        "MathArena: MathArena/aime_2027 (a final-answer competition, 2027) isn't in the suite yet",
        "MathArena: MathArena/arxivmath-0626 (research-level final answers, 2026) isn't in the suite yet"]
