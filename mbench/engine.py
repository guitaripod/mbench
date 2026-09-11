import asyncio
import hashlib
import json
import re
import time

import openai
from openai import AsyncOpenAI

from . import paths, scoring, tool_cases

CONTEXT_ERROR_WORDS = ("context", "too long", "maximum", "exceeds", "max_tokens")
MIN_ANSWER_TOKENS = 2048


def resolve_effort(profile, requested):
    """Turns "max"/"min" into the model's own top or bottom level from its declared effort list; a named level must be one it declares."""
    levels = profile.efforts
    if requested in ("max", "min"):
        if not levels:
            raise ValueError(f"{profile.id} declares no effort levels; add efforts = [...] to its entry in models.toml")
        return levels[-1] if requested == "max" else levels[0]
    if levels and requested not in levels:
        raise ValueError(f"{profile.id} has no '{requested}' effort; it takes {', '.join(levels)} (or max/min)")
    return requested


def supports_seed(profile):
    """SGLang's FlashInfer sampler asserts on seeded top-k/top-p requests and would take the server down, so only llama.cpp and vLLM get seeds."""
    return profile.engine in ("llama.cpp", "vllm")


def seed_for(item_id):
    return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) % 2_147_483_647


def request_kwargs(profile, effort, *, greedy=False, seed=None):
    """How each chat template takes a thinking level; sampling stays on the server's model defaults unless greedy."""
    kwargs, extra = {}, {}
    if profile.thinking == "openai":
        kwargs["reasoning_effort"] = effort
    elif profile.thinking == "qwen":
        extra["chat_template_kwargs"] = {"enable_thinking": True, "reasoning_effort": effort}
    if greedy:
        kwargs["temperature"] = 0
    if seed is not None and supports_seed(profile):
        kwargs["seed"] = seed
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


def injection(profile, effort):
    """The thinking setting as raw request fields, for clients like lmx that can't pass one themselves."""
    kwargs = request_kwargs(profile, effort)
    extra = kwargs.pop("extra_body", {})
    return {**kwargs, **extra}


def context_error(error):
    """A request the server refused because prompt plus answer don't fit its window, which scores zero rather than failing."""
    return isinstance(error, openai.BadRequestError) and any(word in str(error).lower() for word in CONTEXT_ERROR_WORDS)


def refit(error, item):
    """The item again with an answer budget that fits, read off the server's refusal (vLLM and SGLang state the window and
    the prompt's size, llama.cpp sends n_ctx and n_prompt_tokens); None when even the prompt alone doesn't leave room."""
    body = getattr(error, "body", None)
    body = body.get("error", body) if isinstance(body, dict) else {}
    window, prompt = (body or {}).get("n_ctx"), (body or {}).get("n_prompt_tokens")
    text = str(error)
    if window is None:
        match = re.search(r"maximum context length (?:is |of )?(\d+)", text)
        window = int(match.group(1)) if match else None
    if prompt is None:
        match = re.search(r"(\d+) (?:tokens )?(?:from|in) the (?:input )?messages", text)
        prompt = int(match.group(1)) if match else None
    if window is None or prompt is None:
        return None
    room = int(window) - int(prompt) - 64
    if room < MIN_ANSWER_TOKENS or room >= item["max_tokens"]:
        return None
    return {**item, "max_tokens": room}


def base_record(task, item, finish, started):
    return {"id": item["id"], "task": task, "sample": item["sample"], "gold": item["gold"], "meta": item["meta"],
            "finish": finish, "prompt_tokens": None, "completion_tokens": None, "latency": round(time.time() - started, 2),
            "content": "", "reasoning": ""}


def out_of_context(task, item, started):
    return {**base_record(task, item, "context", started), "prediction": None, "score": 0.0}


def build_record(task, item, response, started):
    choice = response.choices[0]
    message = choice.message
    content = message.content or ""
    record = {
        **base_record(task, item, choice.finish_reason, started),
        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
        "completion_tokens": response.usage.completion_tokens if response.usage else None,
        "content": content,
        "reasoning": getattr(message, "reasoning_content", None) or "",
    }
    if task == "lcb":
        record.update(prediction=None, score=None)
    else:
        prediction, score = scoring.TEXT_SCORERS[task](content, item["gold"])
        record.update(prediction=prediction, score=float(score))
    return record


def parse_arguments(raw):
    try:
        arguments = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return arguments if isinstance(arguments, dict) else None


class QualityRunner:
    """Runs one task's items with a fixed number in flight, appending each answer so an interrupted run resumes where it stopped."""

    def __init__(self, profile, run_dir, effort, concurrency, report, abort):
        self.profile = profile
        self.run_dir = run_dir
        self.effort = effort
        self.concurrency = concurrency
        self.report = report
        self.abort = abort
        self.client = AsyncOpenAI(base_url=paths.SWAP_URL + "/v1", api_key="none", timeout=7200, max_retries=0)

    def done_ids(self, task):
        path = self.run_dir / f"{task}.jsonl"
        if not path.exists():
            return set()
        with path.open() as handle:
            return {json.loads(line)["id"] for line in handle if line.strip()}

    async def call(self, item, messages=None, seed_key=None):
        kwargs = request_kwargs(self.profile, self.effort, seed=seed_for(seed_key or item["id"]))
        if item.get("tools"):
            kwargs.update(tools=item["tools"], tool_choice="auto")
        return await self.client.chat.completions.create(
            model=self.profile.id, messages=messages or item["messages"], max_tokens=item["max_tokens"], **kwargs
        )

    async def episode(self, item, started):
        """One tool-use episode: the model acts on a fresh simulated workspace until it answers without a tool call."""
        world = tool_cases.World(tool_cases.BY_ID[item["gold"]]["world"])
        messages = list(item["messages"])
        calls, thoughts = [], []
        prompt_tokens = completion_tokens = malformed = 0
        reply, finish, steps = "", "steps", 0
        for step in range(item["meta"]["max_steps"]):
            response = await self.call(item, messages, f"{item['id']}-{step}")
            steps += 1
            choice = response.choices[0]
            message = choice.message
            if response.usage:
                prompt_tokens += response.usage.prompt_tokens or 0
                completion_tokens += response.usage.completion_tokens or 0
            thought = getattr(message, "reasoning_content", None) or ""
            thoughts.append(thought)
            if not message.tool_calls:
                reply, finish = message.content or "", choice.finish_reason
                break
            assistant = {"role": "assistant", "content": message.content or "", "tool_calls": []}
            if thought:
                assistant["reasoning_content"] = thought
            results = []
            for index, tool_call in enumerate(message.tool_calls):
                call_id = tool_call.id or f"call_{step}_{index}"
                name, raw = tool_call.function.name, tool_call.function.arguments
                assistant["tool_calls"].append({"id": call_id, "type": "function",
                                                "function": {"name": name, "arguments": raw or "{}"}})
                arguments = parse_arguments(raw)
                if arguments is None:
                    malformed += 1
                    result = {"error": "arguments are not a valid JSON object"}
                else:
                    result = world.call(name, arguments)
                calls.append({"name": name, "arguments": arguments or {}, "result": result})
                results.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
            messages += [assistant, *results]
        passed = finish != "steps" and tool_cases.grade(item["gold"], world, calls, reply)
        return {
            **base_record("tools", item, finish, started),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "content": reply, "reasoning": "\n\n".join(thought for thought in thoughts if thought),
            "prediction": calls, "steps": steps, "malformed": malformed, "score": float(passed),
        }

    async def solve(self, task, item, started):
        if task == "tools":
            return await self.episode(item, started)
        return build_record(task, item, await self.call(item), started)

    async def retry_fitted(self, task, item, error, started):
        """A request refused for its length gets one more try with a smaller answer budget, then scores zero."""
        fitted = refit(error, item)
        if fitted is not None:
            try:
                return await self.solve(task, fitted, started)
            except Exception as second:
                if not context_error(second):
                    raise
        return out_of_context(task, item, started)

    async def run(self, task, items):
        out = self.run_dir / f"{task}.jsonl"
        done = self.done_ids(task)
        counter = {"done": len(done)}
        self.report(task, counter["done"], len(items))
        semaphore = asyncio.Semaphore(self.concurrency)
        lock = asyncio.Lock()
        failures = []

        async def one(item):
            async with semaphore:
                if self.abort.is_set():
                    return
                started = time.time()
                if item["meta"].get("unreachable"):
                    record = out_of_context(task, item, started)
                else:
                    try:
                        record = await self.solve(task, item, started)
                    except Exception as error:
                        if not context_error(error):
                            failures.append({"id": item["id"], "error": repr(error)[:500]})
                            return
                        record = await self.retry_fitted(task, item, error, started)
                async with lock:
                    with out.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    counter["done"] += 1
                    self.report(task, counter["done"], len(items))

        pending = [item for item in items if item["id"] not in done]
        await asyncio.gather(*(one(item) for item in pending))
        if failures and not self.abort.is_set():
            retry_ids = {failure["id"] for failure in failures}
            failures.clear()
            await asyncio.sleep(15)
            await asyncio.gather(*(one(item) for item in pending if item["id"] in retry_ids))
        return failures
