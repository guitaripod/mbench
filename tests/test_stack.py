import json
import subprocess

from mbench import board, gpu, stack


def venv(tmp_path, package, version, editable=None):
    site = tmp_path / "lib" / "python3.12" / "site-packages" / f"{package}-{version}.dist-info"
    site.mkdir(parents=True)
    (tmp_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
    if editable:
        (site / "direct_url.json").write_text(json.dumps({"url": f"file://{editable}", "dir_info": {"editable": True}}))
    return tmp_path


def checkout(tmp_path):
    directory = tmp_path / "src"
    (directory / "python").mkdir(parents=True)
    for command in (["init", "-b", "master"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t",
                                                              "commit", "-m", "first", "--allow-empty"]):
        subprocess.run(["git", "-C", str(directory), *command], capture_output=True, check=True)
    return directory


def test_a_reported_version_is_the_build():
    assert stack.build("sglang", {"info": {"version": "0.5.4"}}) == {"engine": "sglang", "version": "0.5.4"}
    assert stack.build("llama.cpp", {"info": {"build_info": "b6420-abc1234"}}) == {"engine": "llama.cpp", "build": "b6420-abc1234"}


def test_a_source_install_is_named_by_its_checkout(tmp_path, monkeypatch):
    source = checkout(tmp_path)
    environment = venv(tmp_path / "venv", "sglang", "0.0.0", editable=source / "python")
    monkeypatch.setattr(stack, "server_pids", lambda: [4242])
    monkeypatch.setattr(stack, "venv_of", lambda pid: environment)
    found = stack.build("sglang", {"info": {"version": "0.0.0"}})
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    assert found["commit"] == head.stdout.strip() and found["source"] == str(source / "python")
    assert stack.build_label(found) == f"SGLang {head.stdout.strip()}"


def test_the_virtualenv_comes_from_the_servers_own_argv(tmp_path, monkeypatch):
    environment = venv(tmp_path / "venv", "sglang", "0.0.0")
    monkeypatch.setattr(gpu, "cmdline", lambda pid: f"{environment}/bin/python3 {environment}/bin/sglang serve --tp 1")
    assert stack.venv_of(1) == environment
    monkeypatch.setattr(gpu, "cmdline", lambda pid: "/usr/bin/python3 -m vllm.entrypoints.openai.api_server")
    assert stack.venv_of(1) is None


def test_a_pip_installed_engine_falls_back_to_the_version_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(stack, "server_pids", lambda: [1])
    monkeypatch.setattr(stack, "venv_of", lambda pid: venv(tmp_path, "vllm", "0.11.2"))
    assert stack.build("vllm", {"info": {}}) == {"engine": "vllm", "version": "0.11.2"}


def test_an_unknown_build_says_only_the_engine(monkeypatch):
    monkeypatch.setattr(stack, "server_pids", lambda: [])
    assert stack.build("llama.cpp", {}) == {"engine": "llama.cpp"}
    assert stack.build_label({"engine": "llama.cpp"}) == "llama.cpp"
    assert stack.build_label({}) is None


def test_a_moved_card_or_build_is_reported_and_an_unrecorded_one_is_not():
    hardware = {"gpu": "RTX PRO 6000", "driver": "610.57.04"}
    build = {"engine": "sglang", "commit": "bbbb"}
    previous = {"hardware": hardware, "server": {"build": {"engine": "sglang", "commit": "aaaa"}}}
    assert stack.changed(previous, hardware, build) == ["SGLang aaaa → SGLang bbbb"]
    older = {"hardware": {"gpu": "RTX 4090", "driver": "570.1"}, "server": {}}
    assert stack.changed(older, hardware, build) == ["RTX 4090 · driver 570.1 → RTX PRO 6000 · driver 610.57.04"]
    assert stack.changed({"hardware": {}, "server": {}}, hardware, build) == []
    assert stack.changed(None, hardware, build) == []


def test_a_runs_stack_is_its_card_and_its_server():
    entry = stack.of({"hardware": {"gpu": "RTX PRO 6000", "driver": "610.57.04", "vram_mib": 97887},
                      "server": {"build": {"engine": "sglang", "commit": "c0ead58"}}})
    assert entry["host"] == "RTX PRO 6000|610.57.04" and entry["build"] == "SGLang c0ead58"
    assert stack.of({})["build"] is None


def test_the_board_groups_models_by_the_setup_that_measured_them():
    models = [{"stack": stack.of({"hardware": {"gpu": "A", "driver": "1"}})},
              {"stack": stack.of({"hardware": {"gpu": "B", "driver": "1"}})},
              {"stack": stack.of({"hardware": {"gpu": "A", "driver": "1"}})}]
    found = board.hosts(models)
    assert [(entry["label"], entry["models"]) for entry in found] == [("A · driver 1", 2), ("B · driver 1", 1)]
