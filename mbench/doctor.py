import json

from openai import AsyncOpenAI

from . import datasets, engine, paths, suite, tool_cases

WEATHER_QUESTION = "What's the weather in Helsinki right now? Use celsius."
TEXT_TOOL_MARKERS = ("get_weather", "<tool_call>", "<|call|>", "[TOOL_CALLS]")


def longest_prompt():
    return max(suite.SUITES["full"]["mrcr"]["bins"])


def outcome(check, status, detail):
    return {"check": check, "status": status, "detail": detail}


def tokens(value):
    return f"{value // 1024}k" if value else "?"


def check_capacity(context, capacity):
    detail = (f"{capacity.get('slots') or '?'} requests at once, {tokens(context)} tokens per request, "
              f"{tokens(capacity.get('pool'))} shared")
    if context and context < longest_prompt():
        return outcome("capacity", "warn", f"{detail}; prompts past {tokens(context)} (MRCR at 128k, the longest speed "
                                           "tests) will score zero")
    return outcome("capacity", "ok", detail)


async def ask(client, profile, effort, messages, max_tokens, tools=None):
    kwargs = engine.request_kwargs(profile, effort)
    if tools:
        kwargs.update(tools=tools, tool_choice="auto")
    return await client.chat.completions.create(model=profile.id, messages=messages, max_tokens=max_tokens, **kwargs)


async def check_answer(client, profile, effort):
    try:
        response = await ask(client, profile, effort,
                             [{"role": "user", "content": "What is 12 times 12? Reply with just the number."}], 4096)
    except Exception as error:
        return outcome("answer", "fail", f"a plain question failed: {error!r}"[:300])
    message = response.choices[0].message
    content, reasoning = message.content or "", getattr(message, "reasoning_content", None) or ""
    if "<think>" in content or "</think>" in content:
        return outcome("answer", "fail", "the reasoning comes back inside the answer; start the server with a reasoning "
                                         "parser (llama.cpp --reasoning-format, SGLang --reasoning-parser)")
    if "144" not in content:
        return outcome("answer", "warn", f"expected 144, got {content[:80]!r} "
                                         f"(finish reason {response.choices[0].finish_reason})")
    if profile.thinking != "none" and effort != engine.EFFORT_OFF and not reasoning:
        return outcome("answer", "warn", "no reasoning came back beside the answer; thinking may be off")
    return outcome("answer", "ok", "answers" + (", reasoning kept apart" if reasoning else ""))


async def check_tool_call(client, profile, effort):
    messages = [{"role": "system", "content": tool_cases.SYSTEM}, {"role": "user", "content": WEATHER_QUESTION}]
    try:
        response = await ask(client, profile, effort, messages, 4096, tool_cases.TOOLS)
    except Exception as error:
        return outcome("tool call", "fail", f"a request with tools failed: {error!r}"[:300]), None
    message = response.choices[0].message
    call = next(iter(message.tool_calls or []), None)
    if call and call.function.name == "get_weather" and engine.parse_arguments(call.function.arguments) is not None:
        return outcome("tool call", "ok", "calls come back structured"), message
    text = message.content or ""
    if any(marker in text for marker in TEXT_TOOL_MARKERS):
        return outcome("tool call", "fail", "the tool call came back as text; start the server with a tool-call parser "
                                            "(llama.cpp --jinja, SGLang --tool-call-parser)"), None
    return outcome("tool call", "warn", f"the model answered without calling get_weather: {text[:80]!r}"), None


async def check_tool_result(client, profile, effort, message):
    """Sends the call back with its result, the way every tool episode continues, which some servers refuse."""
    if message is None:
        return outcome("tool result", "warn", "skipped: there was no tool call to answer")
    call = message.tool_calls[0]
    call_id = call.id or "call_0"
    assistant = {"role": "assistant", "content": message.content or "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}]}
    if getattr(message, "reasoning_content", None):
        assistant["reasoning_content"] = message.reasoning_content
    result = tool_cases.World().call("get_weather", engine.parse_arguments(call.function.arguments))
    messages = [{"role": "system", "content": tool_cases.SYSTEM}, {"role": "user", "content": WEATHER_QUESTION}, assistant,
                {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)}]
    try:
        response = await ask(client, profile, effort, messages, 4096, tool_cases.TOOLS)
    except Exception as error:
        return outcome("tool result", "fail", f"the server refused a conversation carrying a tool result: {error!r}"[:300])
    reply = response.choices[0].message.content or ""
    if str(result.get("temperature")) in reply:
        return outcome("tool result", "ok", "answers from the tool's result")
    return outcome("tool result", "warn", f"the reply didn't use the tool's result: {reply[:80]!r}")


async def check_long_prompt(client, profile, effort, context):
    """One of MRCR's shortest conversations, to see prefill of a 16k-token multi-turn prompt through end to end."""
    items = datasets.mrcr({"bins": (16384,), "per_bin": 1, "max_tokens": 1024}, context)
    if not items or items[0]["meta"].get("unreachable"):
        return outcome("long prompt", "warn", "skipped: a 16k-token prompt doesn't fit this server's context")
    try:
        response = await ask(client, profile, effort, items[0]["messages"], items[0]["max_tokens"])
    except Exception as error:
        return outcome("long prompt", "fail", f"a {tokens(items[0]['meta']['tokens'])}-token conversation failed: "
                                              f"{error!r}"[:300])
    return outcome("long prompt", "ok", f"a {tokens(items[0]['meta']['tokens'])}-token conversation answered "
                                        f"({response.usage.prompt_tokens if response.usage else '?'} prompt tokens)")


def check_stack(moved):
    """Speed, throughput and energy only compare while the card, its driver and the server build hold still."""
    return outcome("stack", "warn", "changed since this model's last run: " + "; ".join(moved)
                   + "; its earlier speed and energy numbers no longer compare")


async def run(profile, effort, context, capacity, moved=()):
    """The checks that decide whether a run can be worth anything, fast enough to go before every run."""
    client = AsyncOpenAI(base_url=(profile.base_url or paths.SWAP_URL) + "/v1", api_key="none", timeout=900,
                         max_retries=0)
    checks = [*([check_stack(moved)] if moved else []),
              check_capacity(context, capacity), await check_answer(client, profile, effort)]
    tool, message = await check_tool_call(client, profile, effort)
    checks += [tool, await check_tool_result(client, profile, effort, message)]
    checks.append(await check_long_prompt(client, profile, effort, context))
    return checks


def failures(checks):
    return [check for check in checks if check["status"] == "fail"]
