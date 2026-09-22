import json
import time

from mbench import board, doctor, hosts, metrics, phone, profiles, stack, store, suite, units

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
    rows = [{"index": index, "decode_tps": tps, "start": 100 + index * 10, "thermal": thermal,
             "completion_tokens": 512}
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
    assert found["speed.tokens_to_knee"]["value"] == 1024


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


def test_the_verdict_is_measured_against_reading_speed():
    def judge(steady, knee):
        return metrics.verdict({"speed.decode_steady": {"value": steady}, "speed.tokens_to_knee": {"value": knee}})

    assert judge(42.0, 2560) == "holds"
    assert judge(29.6, 2048) == "holds"
    assert judge(19.2, 1536) == "fades"
    assert judge(9.3, 512) == "fades"
    assert judge(6.0, 4096) == "barely"
    assert judge(40.0, 512) == "fades"


def test_the_verdict_discriminates_between_the_models_measured():
    measured = {"qwen3-0.6b": 42.1, "lfm2-1.2b": 29.6, "gemma3-1b": 28.6,
                "qwen3-1.7b": 19.2, "qwen3.5-2b": 17.0, "qwen3-4b": 9.3}
    verdicts = {metrics.verdict({"speed.decode_steady": {"value": value},
                                 "speed.tokens_to_knee": {"value": 2048}}) for value in measured.values()}
    assert len(verdicts) > 1, "a verdict every model shares is not a measurement"


def test_a_hot_phone_alone_never_decides_the_verdict():
    hot = {"thermal.worst": {"value": 3}, "thermal.share_hot": {"value": 90.0}}
    assert metrics.verdict({**hot, "speed.decode_steady": {"value": 24.0},
                            "speed.tokens_to_knee": {"value": 2048}}) == "holds"
    assert metrics.verdict(hot) is None


def test_a_run_with_no_thermal_numbers_gets_no_verdict():
    assert metrics.verdict({"index.quality": {"value": 50.0}}) is None


def test_the_board_carries_the_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr("mbench.paths.DB", tmp_path / "bench.db")
    db = store.connect()
    phone_run(db, "air", "qwen3-4b-air", {})
    store.set_metrics(db, "air", {"speed.decode_steady": {"value": 5.0, "unit": "tok/s", "n": 7},
                                  "speed.sustain_ratio": {"value": 0.4, "unit": "x", "n": 20},
                                  "speed.tokens_to_knee": {"value": 512.0, "unit": "tokens", "n": 1},
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


def running(run_id, klass):
    return {"id": run_id, "status": "running", "hardware": {"class": klass} if klass else {}}


def test_a_phone_run_and_a_card_run_do_not_wait_for_each_other():
    runs = [running("air", "phone"), {"id": "queued-gpu", "status": "scheduled", "hardware": {}}]
    assert [run["id"] for run in units.busy(runs, "phone")] == ["air"]
    assert units.busy(runs, "gpu") == []
    assert units.device_class({"hardware": {}}) == "gpu"
    assert units.device_class({"hardware": {"class": "phone"}}) == "phone"


def test_two_runs_on_the_same_device_still_queue():
    runs = [running("one", "gpu"), running("two", "phone")]
    assert [run["id"] for run in units.busy(runs, "gpu")] == ["one"]
    assert [run["id"] for run in units.busy(runs, "phone")] == ["two"]


def test_a_run_that_started_warm_says_so():
    cooled = metrics.cooldown_metrics([{"waited_s": 62.0, "reached": True}, {"waited_s": 71.0, "reached": True}])
    assert cooled["speed.cooldown_s"]["value"] == 133.0 and cooled["speed.cooled"]["value"] == 100.0
    warm = metrics.cooldown_metrics([{"waited_s": 900.0, "reached": False}])
    assert warm["speed.cooled"]["value"] == 0.0


def test_a_card_never_waits_to_cool(monkeypatch):
    monkeypatch.setattr(profiles, "swap_config", lambda: {"models": {"m": {"cmd": "llama-server -m x"}}})
    monkeypatch.setattr(profiles, "user_profiles", lambda: {})
    monkeypatch.setattr(profiles, "omp_overrides", lambda: {})
    assert hosts.for_profile(profiles.resolve("m")).cooldown() == {}
    assert metrics.cooldown_metrics([]) == {}


def test_the_phone_waits_until_it_is_cold_again(monkeypatch):
    states = ["serious", "serious", "fair", "nominal"]
    device = FakeDevice()
    monkeypatch.setattr(phone.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(device, "health", lambda: {"telemetry": {"thermal_state": states.pop(0) if states else "nominal",
                                                                "battery_level": 0.9, "battery_state": "charging"}})
    found = device.cooldown(floor=0, hold=0)
    assert found["reached"] and found["exit"] == "nominal" and found["entry"] == "serious"


def test_the_phone_must_hold_its_cold_state_not_just_touch_it(monkeypatch):
    device = FakeDevice()
    clock = iter([0, 0, 1, 2, 3, 200])
    monkeypatch.setattr(phone.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(phone.time, "time", lambda: next(clock, 200))
    monkeypatch.setattr(device, "health", lambda: {"telemetry": {"thermal_state": "nominal"}})
    assert device.cooldown(floor=0, hold=60, cap=1000)["reached"]


def test_the_phone_gives_up_cooling_and_records_that_it_did(monkeypatch):
    device = FakeDevice()
    clock = iter([0, 0, 0, 10_000])
    monkeypatch.setattr(phone.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(phone.time, "time", lambda: next(clock, 10_000))
    monkeypatch.setattr(device, "health", lambda: {"telemetry": {"thermal_state": "serious"}})
    found = device.cooldown(floor=0, cap=60, hold=0)
    assert not found["reached"] and found["exit"] == "serious"


def test_quality_is_not_inherited_across_two_different_files(tmp_path, monkeypatch):
    monkeypatch.setattr("mbench.paths.DB", tmp_path / "bench.db")
    db = store.connect()
    store.insert_run(db, {"id": "desktop", "model": "m", "name": "m", "suite": suite.label("full"), "effort": "max",
                          "status": "complete", "finished": 1, "hardware": {"gpu": "RTX PRO 6000"}, "profile": {},
                          "server": {"weights": "aaa"}})
    store.set_metrics(db, "desktop", {"index.quality": {"value": 58.4, "unit": "%", "n": 6},
                                      "math.score": {"value": 71.0, "unit": "%", "n": 30}})
    store.insert_run(db, {"id": "other", "model": "m-air", "name": "m-air", "suite": suite.label("phone"),
                          "effort": "max", "status": "complete", "finished": 2, "profile": {},
                          "hardware": {"class": "phone", "device": "iPhone Air"},
                          "server": {"weights": "bbb"}, "flags": {"quality_from": "desktop"}})
    assert board.collect(db)["rankings"]["max"][0]["index"] is None


def test_matching_weights_are_marked_verified(tmp_path, monkeypatch):
    monkeypatch.setattr("mbench.paths.DB", tmp_path / "bench.db")
    db = store.connect()
    store.insert_run(db, {"id": "desktop", "model": "m", "name": "m", "suite": suite.label("full"), "effort": "max",
                          "status": "complete", "finished": 1, "hardware": {"gpu": "RTX PRO 6000"}, "profile": {},
                          "server": {"weights": "aaa"}})
    store.set_metrics(db, "desktop", {"index.quality": {"value": 58.4, "unit": "%", "n": 6},
                                      "math.score": {"value": 71.0, "unit": "%", "n": 30}})
    store.insert_run(db, {"id": "air", "model": "m-air", "name": "m-air", "suite": suite.label("phone"),
                          "effort": "max", "status": "complete", "finished": 2, "profile": {},
                          "hardware": {"class": "phone", "device": "iPhone Air"},
                          "server": {"weights": "aaa"}, "flags": {"quality_from": "desktop"}})
    entry = board.collect(db)["rankings"]["max"][0]
    assert entry["index"]["value"] == 58.4 and entry["qualityFrom"]["verified"] is True


def test_a_digest_is_the_file_not_its_name(tmp_path, monkeypatch):
    monkeypatch.setattr("mbench.paths.CACHE", tmp_path / "cache")
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"one")
    first = stack.digest_of(weights)
    assert first and stack.digest_of(weights) == first
    weights.write_bytes(b"two")
    assert stack.digest_of(weights) != first
    assert stack.digest_of(tmp_path / "missing.gguf") is None


def sustain_rows(speeds):
    return [{"index": index, "decode_tps": tps, "start": 100 + index * 10, "completion_tokens": 512}
            for index, tps in enumerate(speeds)]


def test_stability_is_the_slowest_answer_over_the_fastest():
    found = metrics.sustain_metrics(sustain_rows([40.0, 39.0, 30.0, 24.0, 22.0, 22.0, 22.0, 22.0, 22.0]))
    assert round(found["speed.stability"]["value"], 1) == 55.0
    assert round(found["speed.degradation"]["value"], 1) == 45.0
    assert found["speed.plateau_cv"]["value"] == 0.0


def test_a_device_that_never_throttles_clears_the_bar():
    found = metrics.sustain_metrics(sustain_rows([200.0, 199.0, 200.0, 198.0, 199.0, 200.0, 199.0, 198.0, 199.0]))
    assert found["speed.stability"]["value"] >= metrics.STABLE_PERCENT
    assert found["speed.plateau_cv"]["value"] < 1.0


def test_battery_reads_as_hours_on_a_charge():
    found = metrics.battery_metrics(timeline([(0, "nominal", 1.0), (1800, "fair", 0.9)]))
    assert round(found["battery.drain"]["value"], 1) == 10.0
    assert round(found["battery.hours"]["value"], 2) == 4.75


def test_two_runs_of_the_same_weights_open_the_same_way():
    assert doctor.agreement({"answer": "2 3 5 7 11"}, {"answer": "2 3 5 7 13"}) == {"shared": 9, "of": 10}
    assert doctor.agreement({"answer": "2 3 5"}, {"answer": "The first"}) == {"shared": 0, "of": 5}
    assert doctor.agreement({"answer": ""}, {"answer": "2 3"}) is None
    assert doctor.agreement(None, None) is None


def test_a_dropped_connection_is_retried_not_reported_as_a_dead_phone(monkeypatch):
    device = phone.Device("http://phone:1", "http://phone:2")
    attempts = []
    monkeypatch.setattr(phone.time, "sleep", lambda seconds: None)

    def post(path, payload, timeout=None):
        attempts.append(path)
        if len(attempts) < 2:
            raise phone.http.client.RemoteDisconnected("closed")
        return {"state": "running", "load_seconds": 3.0}

    monkeypatch.setattr(device, "post", post)
    monkeypatch.setattr(device, "reachable", lambda: True)
    assert device.load({"model": "m"})["load_seconds"] == 3.0
    assert len(attempts) == 2


def test_a_phone_that_really_is_gone_fails_the_run(monkeypatch):
    device = phone.Device("http://phone:1", "http://phone:2")
    monkeypatch.setattr(phone.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(device, "post", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no route")))
    monkeypatch.setattr(device, "reachable", lambda: False)
    try:
        device.load({"model": "m"})
    except phone.Unreachable as error:
        assert "did not load" in str(error)
    else:
        raise AssertionError("a dead phone must fail the run")


def test_unload_waits_for_the_server_to_actually_stop(monkeypatch):
    device = phone.Device("http://phone:1", "http://phone:2")
    states = ["stopping", "stopping", "idle"]
    monkeypatch.setattr(phone.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(device, "post", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(device, "health", lambda: {"server": {"state": states.pop(0) if states else "idle"}})
    device.unload()
    assert states == []


def test_the_thermal_story_ignores_the_time_spent_cooling():
    samples = timeline([(0, "serious", 1.0), (30, "fair", 1.0), (600, "nominal", 1.0),
                        (630, "nominal", 1.0), (660, "fair", 1.0), (690, "serious", 1.0)])
    found = metrics.thermal_metrics(samples, cooldowns=[{"started": 0, "ended": 620}])
    assert found["thermal.to_serious_s"]["value"] == 60.0
    assert round(found["thermal.share_hot"]["value"]) == 33


def test_headroom_is_what_ios_had_left():
    result = {"single": [{"footprint_mib": 3346}],
              "telemetry": [{"t": 1, "available_mib": 900}, {"t": 2, "available_mib": 15}]}
    found = metrics.footprint_metrics(result)
    assert found["speed.headroom"]["value"] == 15
    assert round(found["speed.memory_used"]["value"], 1) == 99.6


def test_a_relaunched_app_means_the_model_was_too_big(monkeypatch):
    class Events:
        def __init__(self):
            self.reasons = []

        def emit(self, phase, **fields):
            self.reasons.append(fields.get("reason"))

    class Halt:
        def __init__(self):
            self.reason = None

        def set(self, reason):
            self.reason = reason

    device = FakeDevice()
    uptimes = iter([{"telemetry": {"uptime": 400.0}}, {"telemetry": {"uptime": 3.0}}])
    monkeypatch.setattr(device, "health", lambda: next(uptimes, {"telemetry": {"uptime": 9.0}}))
    halt, events = Halt(), Events()
    guard = phone.Guard(device, halt, events, interval=0)
    guard.run()
    assert halt.reason == "memory" and "more memory" in events.reasons[0]


def test_a_phone_model_can_name_its_desktop_twin(monkeypatch):
    monkeypatch.setattr(profiles, "user_profiles", lambda: {
        "m-air": {"twin": "m-gguf", "phone": {"n_ctx": 4096}}})
    assert profiles.resolve("m-air").twin == "m-gguf"


def test_quality_follows_the_twin_without_naming_a_run(tmp_path, monkeypatch):
    monkeypatch.setattr("mbench.paths.DB", tmp_path / "bench.db")
    db = store.connect()
    for index, run_id in enumerate(("older", "newer")):
        store.insert_run(db, {"id": run_id, "model": "m-gguf", "name": "m", "suite": suite.label("full"),
                              "effort": "max", "status": "complete", "finished": index, "profile": {},
                              "hardware": {"gpu": "RTX PRO 6000"}, "server": {"weights": "aaa"}})
        store.set_metrics(db, run_id, {"index.quality": {"value": 40.0 + index, "unit": "%", "n": 6},
                                       "math.score": {"value": 50.0, "unit": "%", "n": 30}})
    store.insert_run(db, {"id": "air", "model": "m-air", "name": "m-air", "suite": suite.label("phone"),
                          "effort": "max", "status": "complete", "finished": 3, "profile": {},
                          "hardware": {"class": "phone", "device": "iPhone Air"},
                          "server": {"weights": "aaa"}, "flags": {"quality_from": {"twin": "m-gguf"}}})
    entry = board.collect(db)["rankings"]["max"][0]
    assert entry["qualityFrom"]["model"] == "m-gguf" and entry["index"]["value"] == 41.0


def test_a_wedged_app_is_relaunched_before_the_run_is_given_up(monkeypatch):
    monkeypatch.setattr(profiles, "user_profiles", lambda: {"m": {"phone": {"n_ctx": 4096}}})
    host = hosts.for_profile(profiles.resolve("m"))
    calls = {"launched": 0, "loads": 0}

    def load(request):
        calls["loads"] += 1
        if calls["loads"] == 1:
            raise phone.Unreachable("closed")
        return {"load_seconds": 4.0}

    monkeypatch.setattr(phone, "launch", lambda: calls.__setitem__("launched", calls["launched"] + 1))
    monkeypatch.setattr(host.device, "load", load)
    monkeypatch.setattr(host.device, "cooldown", lambda **kwargs: {})
    assert host.ensure_loaded() == 4.0
    assert calls["launched"] == 2


def quality_only(run_id, klass="gpu"):
    return {"id": run_id, "status": "running", "hardware": {"class": klass} if klass != "gpu" else {},
            "flags": {"tasks": ["supergpqa", "math"]}}


def test_runs_that_score_no_speed_may_share_the_card():
    runs = [quality_only("one"), quality_only("two")]
    assert units.busy(runs, "gpu", speed=False) == []
    assert len(units.busy(runs, "gpu", speed=True)) == 2


def test_a_speed_measurement_still_wants_the_device_to_itself():
    runs = [running("timing", "gpu"), quality_only("scoring")]
    assert [run["id"] for run in units.busy(runs, "gpu", speed=False)] == ["timing"]
    assert units.measures_speed(running("timing", "gpu"))
    assert not units.measures_speed(quality_only("scoring"))


class Refusal(Exception):
    pass


def test_a_template_that_refuses_a_conversation_scores_zero(monkeypatch):
    import openai
    from mbench import engine

    class Refused(openai.BadRequestError):
        def __init__(self):
            self.message = "Unable to generate parser for this template"

        def __str__(self):
            return self.message

    error = Refused()
    assert engine.template_error(error)
    assert not engine.context_error(error)
    record = engine.refused("mrcr", {"id": "x", "sample": 0, "gold": "", "meta": {}, "max_tokens": 4096}, 0.0)
    assert record["score"] == 0.0 and record["finish"] == "template"


def test_a_refused_template_warns_rather_than_abandoning_the_run():
    from mbench import doctor
    checks = [{"check": "long prompt", "status": "fail", "detail": "failed: Unable to generate parser for this template"},
              {"check": "answer", "status": "ok", "detail": "answers"}]
    softened = doctor.soften_template_refusals(checks)
    assert softened[0]["status"] == "warn" and "score zero" in softened[0]["detail"]
    assert doctor.failures(softened) == []


def test_a_model_in_a_resident_group_does_not_unload_its_group_mates():
    config = {"models": {"a": {}, "b": {}, "c": {}},
              "groups": {"twins": {"swap": False, "members": ["a", "b"]}}}
    assert profiles.group_of("a", config) == "twins"
    assert profiles.group_of("c", config) is None


def test_a_swapping_group_is_not_resident():
    config = {"groups": {"one-at-a-time": {"swap": True, "members": ["a"]}}}
    assert profiles.group_of("a", config) is None


def test_a_run_waits_for_room_rather_than_dying_at_question_four_hundred(monkeypatch):
    from mbench import worker

    class Events:
        def __init__(self):
            self.said = []

        def emit(self, phase, **fields):
            self.said.append(fields.get("reason"))

    class Halt:
        def __init__(self, stop=False):
            self.stop = stop

        def is_set(self):
            return self.stop

    free = iter([2.0, 3.0, 20.0])
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(worker.gpu, "mem_available_gb", lambda: next(free, 20.0))
    events = Events()
    assert worker.wait_for_memory(events, Halt())
    assert "waiting for room" in events.said[0]


def test_a_run_gives_up_waiting_if_the_box_never_frees_up(monkeypatch):
    from mbench import worker

    class Events:
        def emit(self, phase, **fields):
            pass

    class Halt:
        def is_set(self):
            return True

    monkeypatch.setattr(worker.gpu, "mem_available_gb", lambda: 1.0)
    assert not worker.wait_for_memory(Events(), Halt())


def test_the_run_with_the_least_written_down_gives_way(tmp_path, monkeypatch):
    from mbench import worker
    monkeypatch.setattr("mbench.paths.DB", tmp_path / "bench.db")
    monkeypatch.setattr("mbench.paths.RUNS", tmp_path / "runs")
    db = store.connect()
    for run_id, written in (("20260914-183712-a-instruct", 9000), ("20260914-183712-z-thinking", 10)):
        store.insert_run(db, {"id": run_id, "model": run_id, "name": run_id, "suite": suite.label("full"),
                              "effort": "max", "status": "running", "profile": {}, "hardware": {}})
        directory = tmp_path / "runs" / run_id
        directory.mkdir(parents=True)
        (directory / "supergpqa.jsonl").write_text("x" * written)
    assert worker.MemoryGuard(None, None, "20260914-183712-z-thinking").gives_way()
    assert not worker.MemoryGuard(None, None, "20260914-183712-a-instruct").gives_way()


def test_a_lone_run_always_gives_way(tmp_path, monkeypatch):
    from mbench import worker
    monkeypatch.setattr("mbench.paths.DB", tmp_path / "bench.db")
    store.connect()
    assert worker.MemoryGuard(None, None, "20260914-100000-a").gives_way()
    assert worker.MemoryGuard(None, None, None).gives_way()


def test_only_one_run_reads_a_question_set_at_a_time(tmp_path, monkeypatch):
    import multiprocessing
    from mbench import datasets
    monkeypatch.setattr("mbench.paths.CACHE", tmp_path)
    order = multiprocessing.Manager().list()

    def hold(index):
        with datasets.one_builder_at_a_time():
            order.append(f"in{index}")
            time.sleep(0.2)
            order.append(f"out{index}")

    workers = [multiprocessing.Process(target=hold, args=(index,)) for index in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert [entry.rstrip("0123456789") for entry in order] == ["in", "out"] * 3


def test_a_metal_compute_error_is_named_as_the_model_not_fitting():
    from mbench import doctor
    checks = [{"check": "long prompt", "status": "fail", "detail": "failed: InternalServerError('Compute error.')"}]
    named = doctor.name_memory_failures(checks, phone=True)
    assert "does not fit at this context" in named[0]["detail"]
    assert doctor.name_memory_failures(checks, phone=False) == checks


def test_only_three_quality_runs_share_the_box_at_once():
    sharing = [quality_only(f"run{index}") for index in range(units.MAX_SHARED_RUNS)]
    assert units.busy(sharing, "gpu", speed=False) == sharing
    assert units.busy(sharing[:-1], "gpu", speed=False) == []


def test_a_speed_run_blocks_sharing_runs_whatever_the_count():
    runs = [quality_only("a"), running("timing", "gpu")]
    assert [run["id"] for run in units.busy(runs, "gpu", speed=False)] == ["timing"]


def test_a_wedged_model_server_is_not_a_healthy_run(monkeypatch):
    device = FakeDevice({**HEALTH, "server": {"state": "running", "model": "m"}})
    monkeypatch.setattr(phone.urllib.request, "urlopen",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no answer")))
    assert device.reachable()
    assert not device.answering()


def test_an_idle_app_counts_as_answering():
    assert FakeDevice({**HEALTH, "server": {"state": "idle"}}).answering()


def test_the_tool_is_found_without_a_shell_path(monkeypatch, tmp_path):
    binary = tmp_path / "pymobiledevice3"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.delenv("MBENCH_PMD", raising=False)
    monkeypatch.setattr(phone.shutil, "which", lambda name: None)
    monkeypatch.setattr(phone, "TOOL_PATHS", (str(binary),))
    assert phone.tool() == str(binary)


def test_a_genuinely_missing_tool_still_says_so(monkeypatch):
    monkeypatch.delenv("MBENCH_PMD", raising=False)
    monkeypatch.setattr(phone.shutil, "which", lambda name: None)
    monkeypatch.setattr(phone, "TOOL_PATHS", ("/nowhere/pymobiledevice3",))
    try:
        phone.tool()
    except RuntimeError as error:
        assert "not installed" in str(error)
    else:
        raise AssertionError("a missing tool must be reported")


def test_a_template_refusal_never_counts_as_a_dead_server():
    import openai
    from mbench import engine

    class Refused(openai.BadRequestError):
        def __init__(self):
            self.message = "Unable to generate parser for this template"

        def __str__(self):
            return self.message

    class Runner(engine.QualityRunner):
        def __init__(self):
            self.streak = 0
            self.gone = None

    runner = Runner()
    for _ in range(20):
        runner.note_failure(Refused())
    assert runner.gone is None and runner.streak == 0
    for _ in range(engine.SERVER_GONE_STREAK):
        runner.note_failure(RuntimeError("connection reset"))
    assert runner.gone is not None


def test_every_refused_item_scores_zero_and_none_of_them_fail_the_task():
    from mbench import engine
    item = {"id": "tools-1", "sample": 0, "gold": "g", "meta": {}, "max_tokens": 4096}
    record = engine.refused("tools", item, 0.0)
    assert record["score"] == 0.0 and record["finish"] == "template" and record["prediction"] is None


def test_a_speed_run_keeps_what_it_has_measured(tmp_path):
    from mbench import speed
    kept = []
    run = speed.SpeedRun.__new__(speed.SpeedRun)
    run.partial = kept.append
    run.keep({"single": [1, 2]})
    assert kept == [{"single": [1, 2]}]
    run.partial = None
    run.keep({"single": []})
    assert len(kept) == 1


def test_a_run_spawned_a_moment_ago_is_not_declared_dead(tmp_path, monkeypatch):
    monkeypatch.setattr("mbench.paths.DB", tmp_path / "bench.db")
    monkeypatch.setattr(units, "unit_active", lambda run_id: False)
    db = store.connect()
    store.insert_run(db, {"id": "fresh", "model": "m", "name": "m", "suite": suite.label("full"), "effort": "max",
                          "status": "queued", "profile": {}, "hardware": {}})
    store.update_run(db, "fresh", started=time.time())
    units.reconcile(db)
    assert store.get_run(db, "fresh")["status"] == "queued"


def test_a_run_whose_unit_has_been_gone_a_while_is_failed(tmp_path, monkeypatch):
    monkeypatch.setattr("mbench.paths.DB", tmp_path / "bench.db")
    monkeypatch.setattr(units, "unit_active", lambda run_id: False)
    db = store.connect()
    store.insert_run(db, {"id": "stale", "model": "m", "name": "m", "suite": suite.label("full"), "effort": "max",
                          "status": "running", "profile": {}, "hardware": {}})
    store.update_run(db, "stale", started=time.time() - units.RECONCILE_GRACE_S - 60)
    units.reconcile(db)
    assert store.get_run(db, "stale")["status"] == "failed"


def test_a_phone_that_ran_out_of_room_fails_instead_of_waiting_for_another_turn():
    from mbench import worker

    class Halt:
        reason = "memory"

    class Host:
        kind = "phone"

    class Card:
        fault = "iOS killed the app and it relaunched: the model needed more memory than the phone would give it"

    fault = worker.halt_fault(Halt(), Host(), Card())
    assert fault and "no room for the model" in fault and "more memory" in fault


def test_a_desktop_short_of_ram_is_only_asked_to_wait():
    from mbench import worker

    class Halt:
        reason = "memory"

    class Host:
        kind = "gpu"

    assert worker.halt_fault(Halt(), Host(), None) is None


def test_a_finished_phone_run_gives_the_model_back():
    from mbench import worker

    class Events:
        def log(self, message):
            pass

    class Host:
        kind = "phone"

        def __init__(self):
            self.unloaded = False

        def unload(self):
            self.unloaded = True

    host = Host()
    worker.free_the_phone(host, Events())
    assert host.unloaded


def test_a_finished_gpu_run_leaves_llama_swap_alone():
    from mbench import worker

    class Events:
        def log(self, message):
            pass

    class Host:
        kind = "gpu"

        def unload(self):
            raise AssertionError("a run sharing the box may still be using what llama-swap holds")

    worker.free_the_phone(Host(), Events())


def test_a_model_folder_is_made_before_its_files_are_pushed(tmp_path, monkeypatch):
    folder = tmp_path / "LFM2.5-2.6B-MLX-4bit"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(b"0" * 2048)
    said = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def record(*args, **kwargs):
        said.append(args)
        return Completed()

    monkeypatch.setattr(phone, "bundle_id", lambda: "com.example.mbenchd")
    monkeypatch.setattr(phone, "run_tool", record)
    assert phone.push(folder) == "Documents/models/LFM2.5-2.6B-MLX-4bit"
    assert said[0][:3] == ("apps", "afc", "com.example.mbenchd")
    pushed = [call[4] for call in said[1:]]
    assert pushed == ["Documents/models/LFM2.5-2.6B-MLX-4bit/config.json",
                      "Documents/models/LFM2.5-2.6B-MLX-4bit/model.safetensors"]


def test_a_folder_weighs_what_its_files_weigh(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "one").write_bytes(b"0" * 100)
    (folder / "two").write_bytes(b"0" * 200)
    assert phone.weight_of(folder) == 300
    assert phone.weight_of(folder / "one") == 100


def test_the_app_is_stopped_by_bundle_id(monkeypatch):
    said = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(phone, "bundle_id", lambda: "com.example.mbenchd")
    monkeypatch.setattr(phone, "run_tool", lambda *args, **kwargs: said.append(args) or Completed())
    assert phone.kill()
    assert said[0] == ("developer", "dvt", "pkill", "--bundle", "com.example.mbenchd")


def test_the_phone_is_told_where_the_run_has_got_to(tmp_path):
    from mbench import worker

    class Host:
        def __init__(self):
            self.reports = []

        def report(self, progress):
            self.reports.append(progress)

    host = Host()
    events = worker.Events(tmp_path)
    events.watched_by(host, {"id": "r1", "model": "m-air", "name": "M · iPhone Air", "suite": "phone/v1"})
    events.progress("speed", 3, 43)

    assert host.reports == [{"run": "r1", "model": "M · iPhone Air", "suite": "phone/v1",
                             "phase": "speed", "done": 3, "total": 43, "note": None}]


def test_a_run_nobody_watches_reports_to_nobody(tmp_path):
    from mbench import worker

    events = worker.Events(tmp_path)
    events.progress("speed", 1, 43)
    assert (tmp_path / "events.jsonl").exists()
