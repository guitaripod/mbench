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


def test_unreachable_items_score_zero_without_a_request(tmp_path):
    seen = []
    item = {"id": "mrcr-131072-x", "task": "mrcr", "messages": [], "gold": {"answer": "", "prefix": "p"},
            "meta": {"bin": 131072, "unreachable": True}, "max_tokens": 16384, "sample": 0, "tools": None}
    asyncio.run(runner(tmp_path, [], seen).run("mrcr", [item]))
    record = json.loads((tmp_path / "mrcr.jsonl").read_text())
    assert (record["finish"], record["score"], seen) == ("context", 0.0, [])
