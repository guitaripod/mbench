from types import SimpleNamespace

from mbench import gpu, swap

PMON = """# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command
# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name
    0       1673   C+G      2      0      -      -      -      -    277      0    kwin_wayland
    0    1865400   C+G     17      1      -      -      -      -    574      0    renderD128 --us
    0    2959442     C     81     40      -      -      -      -   9085      0    Borderlands4.exe
    0    2959442     C     79     38      -      -      -      -   9085      0    Borderlands4.exe
    0     777777     C     95     60      -      -      -      -  60000      0    python
"""
GAME = "S:\\steamapps\\common\\Borderlands 4\\OakGame\\Binaries\\Win64\\Borderlands4.exe"


def test_pmon_rows_follow_the_header():
    rows = gpu.pmon_rows(PMON)
    assert rows[1]["command"] == "renderD128 --us" and rows[1]["sm"] == "17"
    assert rows[2]["fb"] == "9085" and len(rows) == 5


def test_contention_skips_model_servers_and_light_desktop_use(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=PMON, returncode=0))
    arguments = {777777: ["/mnt/sglang/.venv/bin/python", "-m", "sglang", "serve"], 2959442: [GAME]}
    monkeypatch.setattr(gpu, "argv", lambda pid: arguments.get(pid, []))
    monkeypatch.setattr(gpu, "parent_of", lambda pid: 1)
    found = gpu.contention()
    assert [(process["name"], process["sm"], process["mib"]) for process in found] == [("Borderlands4.exe", 80.0, 9085)]
    assert "Borderlands4.exe (80% GPU, 8.9 GB)" == gpu.describe_contention(found)


def test_the_compositor_and_the_x_server_are_never_contention(monkeypatch):
    busy = PMON.replace("      2      0      -      -      -      -    277", "     26      0      -      -      -      -    277")
    busy += "    0       1775     G     42      0      -      -      -      -     16      0    Xwayland\n"
    monkeypatch.setattr(gpu.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=busy, returncode=0))
    arguments = {1673: ["/usr/bin/kwin_wayland", "--wayland-fd", "7"], 1775: ["/usr/bin/Xwayland", ":1", "-rootless"],
                 777777: ["/mnt/sglang/.venv/bin/python", "-m", "sglang", "serve"], 2959442: [GAME]}
    monkeypatch.setattr(gpu, "argv", lambda pid: arguments.get(pid, []))
    monkeypatch.setattr(gpu, "parent_of", lambda pid: 1)
    assert [process["name"] for process in gpu.contention()] == ["Borderlands4.exe"]


def test_a_process_is_named_by_its_executable_not_its_arguments():
    assert gpu.program_name(["/usr/bin/Xwayland", ":1", "-auth", "/run/user/1000/xauth_wlOXNO", "-listenfd", "8"]) == "Xwayland"
    assert gpu.program_name(["/home/marcus/ComfyUI/.venv/bin/python3.12", "main.py", "--listen"]) == "python3.12 main.py"
    assert gpu.program_name(["/usr/bin/python", "-m", "comfy"]) == "python comfy"
    assert gpu.program_name(["python3", "-c", "import torch; open('/tmp/x')"]) == "python3"
    assert gpu.program_name([GAME, "-dx12"]) == "Borderlands4.exe"
    assert gpu.program_name(["/opt/vivaldi/vivaldi-bin --type=gpu-process --render-node-override=/dev/dri/renderD128"]) == "vivaldi-bin"
    assert gpu.program_name(["renderD128 --us"]) == "renderD128"
    assert gpu.program_name([]) == "?"


def test_anything_llama_swap_started_counts_as_a_server(monkeypatch):
    lines = {10: "/usr/bin/python3 -m some.worker", 5: "/home/marcus/.local/bin/llama-swap --config x"}
    parents = {10: 5, 5: 1}
    monkeypatch.setattr(gpu, "cmdline", lambda pid: lines.get(pid, ""))
    monkeypatch.setattr(gpu, "parent_of", lambda pid: parents.get(pid))
    assert gpu.is_server(10)
    assert not gpu.is_server(1)


def test_energy_integrates_board_power_over_time():
    sampler = gpu.Sampler.__new__(gpu.Sampler)
    sampler.samples = [(0.0, 100.0, 0, 0), (1800.0, 300.0, 0, 0), (3600.0, 300.0, 0, 0)]
    assert sampler.energy_wh(0, 3600) == 250.0
    assert sampler.energy_wh(5000, 6000) is None


def test_capacity_from_each_kind_of_server():
    sglang = {"endpoint": "/server_info",
              "info": {"max_running_requests": 4, "context_length": 131072, "max_total_num_tokens": 161216}}
    assert swap.capacity(sglang) == {"slots": 4, "context": 131072, "pool": 161216}
    older = {"endpoint": "/get_server_info", "info": {"max_running_requests": 8, "context_length": 524288,
                                                      "internal_states": [{"max_total_num_tokens": 300000}]}}
    assert swap.capacity(older) == {"slots": 8, "context": 300000, "pool": 300000}
    llama = {"endpoint": "/props", "info": {"total_slots": 4, "default_generation_settings": {"n_ctx": 65536}}}
    assert swap.capacity(llama) == {"slots": 4, "context": 65536, "pool": 65536}
    vllm = {"endpoint": "/v1/models", "info": {"data": [{"id": "m", "max_model_len": 32768}]}}
    assert swap.capacity(vllm) == {"slots": None, "context": 32768, "pool": None}
    assert swap.capacity({}) == {"slots": None, "context": None, "pool": None}
