import json

from mbench import board, hosts, metrics, phone, profiles, stack, store, suite

HEALTH = {
    "app": {"version": "0.1.0", "llama_commit": "5266f24", "llama_build": 50},
    "device": {"hardware": "iPhone18,4", "system": "iOS 27.0", "processors": 6},
    "telemetry": {"thermal_state": "fair", "footprint_mib": 4821.0, "battery_level": 0.97},
    "server": {"state": "running", "model": "qwen3-4b"},
}


class FakeDevice(phone.Device):
    def __init__(self, health=None):
        super().__init__("http://phone:1", "http://phone:2")
        self.payload = health if health is not None else HEALTH
        self.loaded = None

    def health(self):
        return self.payload

    def load(self, request):
        self.loaded = request
        return {"state": "running", "load_seconds": 12.5}


def test_a_known_phone_is_named_and_carries_its_bandwidth():
    described = FakeDevice().describe()
    assert described["class"] == "phone"
    assert described["device"] == "iPhone Air" and described["soc"] == "A19 Pro"
    assert described["bandwidth_gbs"] == 68.3 and described["ram_gb"] == 12


def test_an_unknown_phone_still_records_what_answered():
    described = FakeDevice({**HEALTH, "device": {"hardware": "iPhone99,9", "system": "iOS 30"}}).describe()
    assert described["device"] == "iPhone99,9" and described["soc"] is None


def test_the_app_build_is_the_stack_a_phone_run_records():
    assert FakeDevice().build() == {"engine": "llama.cpp", "version": "b50", "commit": "5266f24", "app": "0.1.0"}


def test_a_phone_host_sends_the_models_toml_settings_as_the_load_request(monkeypatch):
    settings = {"file": "qwen3-4b-q4km", "n_ctx": 32768, "parallel": 4, "control_url": "http://phone:1"}
    monkeypatch.setattr(profiles, "user_profiles", lambda: {"qwen3-4b-air": {"phone": settings}})
    profile = profiles.resolve("qwen3-4b-air")
    host = hosts.for_profile(profile)
    host.device = FakeDevice()
    assert host.ensure_loaded() == 12.5
    assert host.device.loaded == {"model": "qwen3-4b-q4km", "n_ctx": 32768, "parallel": 4}


def test_a_phone_profile_needs_no_llama_swap_entry(monkeypatch):
    monkeypatch.setattr(profiles, "user_profiles", lambda: {
        "qwen3-4b-air": {"name": "Qwen3 4B · iPhone Air", "thinking": "qwen", "phone": {"n_ctx": 32768}}})
    monkeypatch.setattr(profiles, "swap_config", lambda: {"models": {}})
    profile = profiles.resolve("qwen3-4b-air")
    assert profile.context == 32768 and profile.thinking == "qwen"
    assert profile.base_url == phone.SERVER_URL and json.loads(profile.cmd)["model"] == "qwen3-4b-air"


def test_the_load_request_is_the_fingerprint(monkeypatch):
    def resolve(settings):
        monkeypatch.setattr(profiles, "user_profiles", lambda: {"m": {"phone": settings}})
        return profiles.resolve("m").fingerprint

    assert resolve({"n_ctx": 32768}) == resolve({"n_ctx": 32768, "control_url": "http://elsewhere"})
    assert resolve({"n_ctx": 32768}) != resolve({"n_ctx": 16384})


def test_a_phone_host_never_reports_gpu_contention(monkeypatch):
    monkeypatch.setattr(profiles, "user_profiles", lambda: {"m": {"phone": {"n_ctx": 4096}}})
    host = hosts.for_profile(profiles.resolve("m"))
    assert host.kind == "phone" and host.contention() == []
    assert host.sampler().energy_wh(0, 1) is None


def test_the_sampler_reports_the_worst_thermal_state_a_request_ran_under():
    device = FakeDevice()
    sampler = phone.Sampler(device, interval_ms=100_000)
    sampler.samples = [(10, {"thermal_state": "nominal", "footprint_mib": 100, "battery_level": 1.0}),
                       (11, {"thermal_state": "serious", "footprint_mib": 220, "battery_level": 0.99})]
    window = sampler.window(9, 12)
    sampler.close()
    assert window["thermal"] == "serious" and window["thermal_start"] == "nominal"
    assert window["footprint_mib"] == 220 and window["battery"] == 0.995


def test_sustained_decode_reports_what_it_settles_at():
    rows = [{"index": index, "decode_tps": tps, "start": 100 + index * 10, "thermal": thermal}
            for index, (tps, thermal) in enumerate(
                [(40.0, "nominal"), (39.0, "nominal"), (30.0, "fair"), (24.0, "fair"),
                 (23.0, "serious"), (22.0, "serious"), (22.5, "serious"), (22.0, "serious"), (22.0, "serious")])]
    found = metrics.sustain_metrics(rows)
    assert found["speed.decode_peak"]["value"] == 40.0
    assert found["speed.decode_steady"]["value"] == 22.0
    assert round(found["speed.sustain_ratio"]["value"], 3) == 0.55
    assert found["speed.throttle_s"]["value"] == 20.0
    assert found["speed.sustain.02"]["value"] == 30.0
    assert found["speed.sustain_state.04"]["value"] == 2


def test_a_short_sustain_run_reports_nothing():
    assert metrics.sustain_metrics([{"index": 0, "decode_tps": 10.0, "start": 1}]) == {}


def test_a_phone_is_its_own_host_on_the_board():
    hardware = {"class": "phone", "device": "iPhone Air", "soc": "A19 Pro", "os": "iOS 27.0"}
    assert stack.host_label(hardware) == "iPhone Air · A19 Pro · iOS 27.0"
    assert stack.host_key(hardware) != stack.host_key({"gpu": "RTX PRO 6000", "driver": "580.95"})
    assert stack.of({"hardware": hardware})["deviceClass"] == "phone"


def phone_run(db, run_id, model, flags):
    store.insert_run(db, {"id": run_id, "model": model, "name": model, "suite": suite.label("phone"),
                          "effort": "max", "status": "complete", "flags": flags, "finished": 2,
                          "hardware": {"class": "phone", "device": "iPhone Air"}, "profile": {}})


def test_a_phone_run_shows_the_quality_of_the_desktop_run_it_names(tmp_path, monkeypatch):
    monkeypatch.setattr("mbench.paths.DB", tmp_path / "bench.db")
    db = store.connect()
    store.insert_run(db, {"id": "desktop", "model": "qwen3-4b", "name": "Qwen3 4B", "suite": suite.label("full"),
                          "effort": "max", "status": "complete", "finished": 1, "hardware": {"gpu": "RTX PRO 6000"},
                          "profile": {}})
    store.set_metrics(db, "desktop", {"index.quality": {"value": 58.4, "unit": "%", "n": 6},
                                      "math.score": {"value": 71.0, "unit": "%", "n": 30}})
    phone_run(db, "air", "qwen3-4b-air", {"quality_from": "desktop"})
    store.set_metrics(db, "air", {"speed.decode_steady": {"value": 11.9, "unit": "tok/s", "n": 7}})
    data = board.collect(db)
    entry = next(model for model in data["rankings"]["max"] if model["id"] == "qwen3-4b-air")
    assert entry["deviceClass"] == "phone"
    assert entry["index"]["value"] == 58.4 and entry["qualityFrom"]["run"] == "desktop"
    assert entry["tasks"]["math"]["value"] == 71.0
    assert entry["speed"]["decode_steady"]["value"] == 11.9


def test_a_phone_run_without_a_named_quality_run_shows_none(tmp_path, monkeypatch):
    monkeypatch.setattr("mbench.paths.DB", tmp_path / "bench.db")
    db = store.connect()
    phone_run(db, "air", "qwen3-4b-air", {})
    entry = board.collect(db)["rankings"]["max"][0]
    assert entry["index"] is None and entry["tasks"] == {} and entry["qualityFrom"] is None


def timeline(entries, plugged=False):
    return [{"t": float(at), "thermal": state, "battery": level,
             "battery_state": "charging" if plugged else "unplugged"}
            for at, state, level in entries]


def test_a_phone_run_says_how_long_it_stayed_cool_and_how_long_it_ran_hot():
    found = metrics.thermal_metrics(timeline([
        (0, "nominal", 1.0), (30, "nominal", 0.99), (60, "fair", 0.98),
        (90, "serious", 0.97), (120, "serious", 0.96), (150, "serious", 0.95),
    ]), tokens=4000)
    assert found["thermal.worst"]["value"] == 2
    assert found["thermal.to_fair_s"]["value"] == 60.0
    assert found["thermal.to_serious_s"]["value"] == 90.0
    assert round(found["thermal.share_hot"]["value"]) == 50
    assert round(found["battery.drain"]["value"], 1) == 5.0
    assert round(found["battery.per_hour"]["value"]) == 120
    assert round(found["battery.per_1k_tokens"]["value"], 2) == 1.25


def test_a_charging_run_reports_no_battery_cost():
    found = metrics.thermal_metrics(timeline([
        (0, "nominal", 1.0), (30, "fair", 1.0), (60, "fair", 1.0)], plugged=True))
    assert "battery.drain" not in found and found["thermal.worst"]["value"] == 1


def test_a_desktop_run_has_no_thermal_story():
    assert metrics.thermal_metrics([]) == {}
    assert metrics.thermal_metrics(None) == {}


def test_the_verdict_disqualifies_a_model_that_runs_hot_or_crawls():
    def judge(steady, ratio, worst, hot):
        return metrics.verdict({"speed.decode_steady": {"value": steady}, "speed.sustain_ratio": {"value": ratio},
                                "thermal.worst": {"value": worst}, "thermal.share_hot": {"value": hot}})

    assert judge(24.0, 0.92, 1, 0.0) == "holds"
    assert judge(18.0, 0.70, 1, 0.0) == "fades"
    assert judge(18.0, 0.85, 2, 10.0) == "fades"
    assert judge(6.0, 0.95, 0, 0.0) == "barely"
    assert judge(18.0, 0.40, 1, 0.0) == "barely"
    assert judge(18.0, 0.90, 1, 80.0) == "barely"
    assert judge(18.0, 0.90, 3, 0.0) == "barely"


def test_a_run_with_no_thermal_numbers_gets_no_verdict():
    assert metrics.verdict({"index.quality": {"value": 50.0}}) is None


def test_the_board_carries_the_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr("mbench.paths.DB", tmp_path / "bench.db")
    db = store.connect()
    phone_run(db, "air", "qwen3-4b-air", {})
    store.set_metrics(db, "air", {"speed.decode_steady": {"value": 5.0, "unit": "tok/s", "n": 7},
                                  "speed.sustain_ratio": {"value": 0.4, "unit": "x", "n": 20},
                                  "thermal.share_hot": {"value": 70.0, "unit": "%", "n": 20},
                                  "battery.per_hour": {"value": 31.0, "unit": "%/h", "n": 20}})
    entry = board.collect(db)["rankings"]["max"][0]
    assert entry["verdict"] == "barely"
    assert entry["thermal"]["share_hot"]["value"] == 70.0
    assert entry["battery"]["per_hour"]["value"] == 31.0


def test_a_phone_suite_cannot_score_quality_at_all():
    assert set(suite.SUITES["phone"]) == {"speed"}


def test_a_phone_row_only_takes_quality_from_a_full_question_set(tmp_path, monkeypatch):
    monkeypatch.setattr("mbench.paths.DB", tmp_path / "bench.db")
    db = store.connect()
    store.insert_run(db, {"id": "small", "model": "qwen3-0.6b", "name": "Qwen3 0.6B", "suite": suite.label("smoke"),
                          "effort": "max", "status": "complete", "finished": 1, "hardware": {"gpu": "RTX PRO 6000"},
                          "profile": {}})
    store.set_metrics(db, "small", {"index.quality": {"value": 91.0, "unit": "%", "n": 6}})
    phone_run(db, "air", "qwen3-0.6b-air", {"quality_from": "small"})
    entry = board.collect(db)["rankings"]["max"][0]
    assert entry["index"] is None and entry["qualityFrom"] is None
