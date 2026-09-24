import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import openai
import pytest

from mbench import engine, tool_cases
from mbench.engine import QualityRunner, request_kwargs, resolve_effort
from mbench.profiles import Profile


def profile(thinking="qwen", engine_name="sglang", efforts=("low", "medium", "xhigh")):
    return Profile(id="m", name="m", engine=engine_name, thinking=thinking, context=None, efforts=list(efforts))


def test_max_and_min_follow_the_declared_levels():
    assert resolve_effort(profile(), "max") == "xhigh"
    assert resolve_effort(profile(efforts=("minimal", "low", "medium", "high")), "max") == "high"
    assert resolve_effort(profile(efforts=("minimal", "low", "medium", "high")), "min") == "minimal"


def test_undeclared_levels_are_rejected_and_undeclared_models_pass_through():
    with pytest.raises(ValueError, match="takes low, medium, xhigh"):
        resolve_effort(profile(), "high")
    assert resolve_effort(profile(efforts=()), "high") == "high"
    with pytest.raises(ValueError, match="declares no effort levels"):
        resolve_effort(profile(efforts=()), "max")


def test_levels_reach_the_request_unchanged():
    kwargs = request_kwargs(profile(), "xhigh")
    assert kwargs["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    assert request_kwargs(profile("openai"), "low", seed=7) == {"reasoning_effort": "low"}
    assert request_kwargs(profile("none", "llama.cpp"), "medium", seed=7) == {"seed": 7}


def test_muse_takes_its_level_as_reasoning_strength():
    muse = profile("muse", efforts=("low", "medium", "high", "xhigh"))
    assert request_kwargs(muse, "xhigh") == {"extra_body": {"chat_template_kwargs": {"reasoning_strength": "xhigh"}}}
    with pytest.raises(ValueError, match="can't switch thinking off"):
        resolve_effort(muse, "none")


def refusal(message, body=None):
    response = httpx.Response(400, request=httpx.Request("POST", "http://swap/v1/chat/completions"))
    return openai.BadRequestError(message, response=response, body=body)


def test_refusals_for_length_are_refitted_from_the_servers_numbers():
    item = {"id": "x", "max_tokens": 16384}
    sglang = refusal("Requested token count exceeds the model's maximum context length of 131072 tokens. You requested "
                     "a total of 140000 tokens: 123616 tokens from the input messages and 16384 tokens for the completion.")
    assert engine.context_error(sglang)
    assert engine.refit(sglang, item)["max_tokens"] == 131072 - 123616 - 64
    llama = refusal("the request exceeds the available context size", {"error": {"n_ctx": 65536, "n_prompt_tokens": 70000}})
    assert engine.context_error(llama) and engine.refit(llama, item) is None
    assert not engine.context_error(refusal("invalid tool schema"))


def message(content="", tool_calls=None, reasoning=""):
    return SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=reasoning)


def response(msg, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def runner(tmp_path, script, seen):
    instance = QualityRunner(profile(), tmp_path, "medium", lambda *args: None, threading.Event(), slots=2)
    turns = iter(script)

    async def fake_call(item, messages=None, seed_key=None):
        seen.append(json.loads(json.dumps(messages or item["messages"])))
        return next(turns)

    instance.call = fake_call
    return instance


def tools_item(case_id, max_steps=10):
    entry = tool_cases.BY_ID[case_id]
    return {"id": f"tools-{case_id}-r0", "task": "tools", "messages": tool_cases.messages_for(entry), "gold": case_id,
            "meta": {"case": case_id, "category": entry["category"], "max_steps": max_steps}, "max_tokens": 1024,
            "sample": 0, "tools": tool_cases.TOOLS}


def test_an_episode_runs_tools_until_the_model_answers(tmp_path):
    seen = []
    script = [
        response(message(tool_calls=[call("c1", "list_events", {"date": "2026-09-14"})], reasoning="look first"),
                 "tool_calls"),
        response(message(tool_calls=[call("c2", "delete_event", {"event_id": "E1"})]), "tool_calls"),
        response(message("Cancelled your standup on Monday.")),
    ]
    record = asyncio.run(runner(tmp_path, script, seen).episode(tools_item("cancel-one"), 0.0))
    assert record["score"] == 1.0 and record["steps"] == 3 and record["completion_tokens"] == 60
    assert [entry["name"] for entry in record["prediction"]] == ["list_events", "delete_event"]
    final_turn = seen[-1]
    assert final_turn[-4]["tool_calls"][0]["function"]["name"] == "list_events"
    assert final_turn[-4]["reasoning_content"] == "look first"
    assert json.loads(final_turn[-3]["content"])["events"][0]["id"] == "E1"
    assert final_turn[-1] == {"role": "tool", "tool_call_id": "c2", "content": json.dumps({"ok": True, "deleted": "E1"})}


def test_an_episode_that_never_answers_fails(tmp_path):
    script = [response(message(tool_calls=[call(f"c{step}", "list_orders", {})]), "tool_calls") for step in range(3)]
    record = asyncio.run(runner(tmp_path, script, []).episode(tools_item("no-tool-thanks", max_steps=3), 0.0))
    assert record["finish"] == "steps" and record["score"] == 0.0


def test_malformed_arguments_are_reported_back_and_counted(tmp_path):
    bad = SimpleNamespace(id="c1", function=SimpleNamespace(name="get_order", arguments="{order_id: A1002"))
    script = [response(message(tool_calls=[bad]), "tool_calls"), response(message("Sorry, which order?"))]
    seen = []
    record = asyncio.run(runner(tmp_path, script, seen).episode(tools_item("refund-too-old"), 0.0))
    assert record["malformed"] == 1 and record["score"] == 0.0
    assert "not a valid JSON object" in seen[-1][-1]["content"]
    assert seen[-1][-2]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_unreachable_items_score_zero_without_a_request(tmp_path):
    seen = []
    item = {"id": "mrcr-131072-x", "task": "mrcr", "messages": [], "gold": {"answer": "", "prefix": "p"},
            "meta": {"bin": 131072, "unreachable": True}, "max_tokens": 16384, "sample": 0, "tools": None}
    asyncio.run(runner(tmp_path, [], seen).run("mrcr", [item]))
    record = json.loads((tmp_path / "mrcr.jsonl").read_text())
    assert (record["finish"], record["score"], seen) == ("context", 0.0, [])


def items_for(task, count):
    return [{"id": f"{task}-{index}", "task": task, "messages": [], "gold": "A", "meta": {}, "max_tokens": 16,
             "sample": 0, "tools": None} for index in range(count)]


def dead(tmp_path, monkeypatch, failures, total):
    monkeypatch.setattr(engine.gpu, "last_fault", lambda *args: "Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.")
    instance = QualityRunner(profile(), tmp_path, "medium", lambda *args: None, threading.Event(), slots=4)
    calls = {"n": 0}

    async def fake_call(item, messages=None, seed_key=None):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise openai.InternalServerError("upstream command exited prematurely", response=httpx.Response(
                500, request=httpx.Request("POST", "http://swap/v1/chat/completions")), body=None)
        return response(message("Answer: A"))

    instance.call = fake_call
    return instance, items_for("supergpqa", total)


def test_a_server_that_stops_answering_stops_the_run(tmp_path, monkeypatch):
    runner_instance, items = dead(tmp_path, monkeypatch, failures=40, total=20)
    with pytest.raises(engine.ServerGone, match="fallen off the bus"):
        asyncio.run(runner_instance.run("supergpqa", items))


def test_a_few_failures_are_only_failures(tmp_path, monkeypatch):
    runner_instance, items = dead(tmp_path, monkeypatch, failures=2, total=20)
    failures = asyncio.run(runner_instance.run("supergpqa", items))
    assert failures == [] and len(runner_instance.done("supergpqa")) == 20


def unparsed(tmp_path, garbled_ids):
    """A llama.cpp server that refuses the model's output for some questions every time they are asked."""
    instance = QualityRunner(profile(), tmp_path, "medium", lambda *args: None, threading.Event(), slots=4)

    async def fake_call(item, messages=None, seed_key=None):
        if item["id"] in garbled_ids:
            raise openai.InternalServerError(
                "Error code: 500 - The model produced output that does not match the expected peg-native format",
                response=httpx.Response(500, request=httpx.Request("POST", "http://swap/v1/chat/completions")), body=None)
        return response(message("Answer: A"))

    instance.call = fake_call
    return instance


def test_output_the_server_cannot_parse_is_retried_then_scored_zero_not_a_dead_server(tmp_path):
    items = items_for("supergpqa", 20)
    garbled_ids = {item["id"] for item in items[:12]}
    failures = asyncio.run(unparsed(tmp_path, garbled_ids).run("supergpqa", items))
    records = {record["id"]: record for record in map(json.loads, (tmp_path / "supergpqa.jsonl").read_text().splitlines())}
    assert failures == [] and len(records) == 20
    assert all((records[item_id]["finish"], records[item_id]["score"]) == ("unparsed", 0.0) for item_id in garbled_ids)
    assert engine.unparsed_output(openai.InternalServerError(
        "The model produced output that does not match the expected peg-native format",
        response=httpx.Response(500, request=httpx.Request("POST", "http://swap")), body=None))
    assert not engine.unparsed_output(openai.InternalServerError(
        "upstream command exited prematurely",
        response=httpx.Response(500, request=httpx.Request("POST", "http://swap")), body=None))



def test_answers_that_outgrow_a_shared_cache_together_go_again_one_at_a_time(tmp_path):
    instance = QualityRunner(profile(), tmp_path, "medium", lambda *args: None, threading.Event(), slots=4)
    flight = {"now": 0, "most": 0}

    async def fake_call(item, messages=None, seed_key=None):
        flight["now"] += 1
        flight["most"] = max(flight["most"], flight["now"])
        await asyncio.sleep(0.01)
        crowded = flight["now"] > 1
        flight["now"] -= 1
        if crowded:
            raise openai.InternalServerError(
                "Error code: 500 - Context size has been exceeded.",
                response=httpx.Response(500, request=httpx.Request("POST", "http://swap/v1/chat/completions")), body=None)
        return response(message("Answer: A"))

    instance.call = fake_call
    failures = asyncio.run(instance.run("graphwalks", items_for("graphwalks", 4)))
    records = [json.loads(line) for line in (tmp_path / "graphwalks.jsonl").read_text().splitlines()]
    assert failures == [] and flight["most"] == 4 and len(records) == 4
    assert all(record["finish"] == "stop" for record in records)
