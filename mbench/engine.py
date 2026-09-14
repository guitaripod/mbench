import asyncio
import hashlib
import json
import re
import time
from collections import deque

import openai
from openai import AsyncOpenAI

from . import gpu, paths, scoring, suite, tool_cases

CONTEXT_ERROR_WORDS = ("context", "too long", "maximum", "exceeds", "max_tokens")
TEMPLATE_ERROR_WORDS = ("template", "parser", "alternate", "role")
MIN_ANSWER_TOKENS = 2048
ANSWER_RESERVE = 16384
POOL_HEADROOM = 0.9
SERVER_GONE_STREAK = 8
FALLBACK_TOKENS_PER_SECOND = 40
EFFORT_OFF = "none"


def resolve_effort(profile, requested):
    """Turns "max"/"min" into the model's own top or bottom level from its declared effort list; a named level must be one
    it declares, and "none" needs a template that can switch thinking off."""
    levels = profile.efforts
    if requested == EFFORT_OFF:
        if EFFORT_OFF in levels or profile.thinking in ("qwen", "none"):
            return EFFORT_OFF
        raise ValueError(f"{profile.id} can't switch thinking off; its levels are {', '.join(levels) or 'undeclared'}, "
                         "and --effort min is its lowest")
    if requested in ("max", "min"):
        if not levels:
            raise ValueError(f"{profile.id} declares no effort levels; add efforts = [...] to its entry in models.toml")
        return levels[-1] if requested == "max" else levels[0]
    if levels and requested not in levels:
        raise ValueError(f"{profile.id} has no '{requested}' effort; it takes {', '.join(levels)} (or max/min/none)")
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
        extra["chat_template_kwargs"] = ({"enable_thinking": False} if effort == EFFORT_OFF
                                         else {"enable_thinking": True, "reasoning_effort": effort})
    elif profile.thinking == "muse":
        extra["chat_template_kwargs"] = {"reasoning_strength": effort}
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


def template_error(error):
    """A conversation the model's own chat template refuses — Gemma's insists on strict user/model alternation, and
    llama.cpp cannot build a parser for anything else. Every item of that shape will be refused the same way, so it
    is a zero the model earned, not a server that broke."""
    return (isinstance(error, openai.BadRequestError) and not context_error(error)
            and any(word in str(error).lower() for word in TEMPLATE_ERROR_WORDS))


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


def refused(task, item, started):
    return {**base_record(task, item, "template", started), "prediction": None, "score": 0.0}


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


def prompt_tokens(item):
    """The prompt's size: counted for long-context items, estimated from characters for the rest."""
    counted = item["meta"].get("tokens")
    if counted:
        return counted
    return sum(len(str(message.get("content") or "")) for message in item["messages"]) // 3 + 512


def weight(item):
    """Context an item may hold on the server while it runs: its prompt plus room for a typical answer."""
    return prompt_tokens(item) + min(item["max_tokens"], ANSWER_RESERVE)


class Gate:
    """Starts an item once a request slot is free and its context fits the server's shared pool. Waiters start in order,
    so a long prompt is never starved by a stream of short ones; one too big for the pool runs alone."""

    def __init__(self, slots, pool=None):
        self.slots, self.pool = slots, pool
        self.running = self.used = 0
        self.waiters = deque()

    def fits(self, size):
        return self.running < self.slots and (self.pool is None or self.running == 0 or self.used + size <= self.pool)

    def take(self, size):
        self.running += 1
        self.used += size

    async def acquire(self, size):
        if not self.waiters and self.fits(size):
            self.take(size)
            return
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((size, future))
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release(size)
            raise

    def release(self, size):
        self.running -= 1
        self.used -= size
        while self.waiters and (self.waiters[0][1].cancelled() or self.fits(self.waiters[0][0])):
            size, future = self.waiters.popleft()
            if not future.cancelled():
                self.take(size)
                future.set_result(None)


def percentile(values, share):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * (len(ordered) - 1)))]


class ServerGone(RuntimeError):
    """The server stopped answering part way through a task. Those answers are missing, not wrong, so the run stops
    rather than scoring a dead server's silence."""


class QualityRunner:
    """Runs one task's items as many at a time as the server takes, appending each answer so an interrupted run resumes
    where it stopped. It stops dispatching when an item couldn't finish before the deadline, and stops outright when
    told to (low RAM, the GPU needed elsewhere)."""

    def __init__(self, profile, run_dir, effort, report, stop, slots=None, pool=None, deadline=None):
        self.profile = profile
        self.run_dir = run_dir
        self.effort = effort
        self.report = report
        self.stop = stop
        self.slots = max(1, min(slots or suite.CONCURRENCY, suite.MAX_CONCURRENCY))
        self.pool = int(pool * POOL_HEADROOM) if pool else None
        self.deadline = deadline
        self.drained = False
        self.gone = None
        self.streak = 0
        self.latencies = {}
        self.client = AsyncOpenAI(base_url=(profile.base_url or paths.SWAP_URL) + "/v1", api_key="none", timeout=7200,
                                  max_retries=0)

    def done(self, task):
        path = self.run_dir / f"{task}.jsonl"
        if not path.exists():
            return []
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def expected_seconds(self, task, item):
        """How long an item takes: the slow end of this task's answers so far, or its token budget at a modest pace."""
        seen = self.latencies.get(task) or []
        if len(seen) >= 3:
            return percentile(seen, 0.9)
        return item["max_tokens"] / FALLBACK_TOKENS_PER_SECOND

    def too_late(self, task, item):
        return self.deadline is not None and time.time() + self.expected_seconds(task, item) > self.deadline

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
        prompt_count = completion_count = malformed = 0
        reply, finish, steps = "", "steps", 0
        for step in range(item["meta"]["max_steps"]):
            response = await self.call(item, messages, f"{item['id']}-{step}")
            steps += 1
            choice = response.choices[0]
            message = choice.message
            if response.usage:
                prompt_count += response.usage.prompt_tokens or 0
                completion_count += response.usage.completion_tokens or 0
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
            "prompt_tokens": prompt_count, "completion_tokens": completion_count,
            "content": reply, "reasoning": "\n\n".join(thought for thought in thoughts if thought),
            "prediction": calls, "steps": steps, "malformed": malformed, "score": float(passed),
        }

    async def solve(self, task, item, started):
        if task == "tools":
            return await self.episode(item, started)
        return build_record(task, item, await self.call(item), started)

    async def retry_fitted(self, task, item, error, started):
        """A request refused for its length gets one more try with a smaller answer budget, then scores zero."""
        if template_error(error):
            return refused(task, item, started)
        fitted = refit(error, item)
        if fitted is not None:
            try:
                return await self.solve(task, fitted, started)
            except Exception as second:
                if not context_error(second):
                    raise
        return out_of_context(task, item, started)

    def note_failure(self, error):
        """A server that has died answers nothing at all, so its failures arrive one after another."""
        self.streak += 1
        if self.streak >= SERVER_GONE_STREAK:
            self.gone = repr(error)[:300]

    def check_alive(self, task, pending, failures):
        """Most of a task failing means the server went away, not that the model got the answers wrong."""
        if not pending or (not self.gone and len(failures) < max(SERVER_GONE_STREAK, len(pending) // 2)):
            return
        fault = gpu.last_fault()
        detail = self.gone or (failures[0]["error"] if failures else "")
        raise ServerGone(f"the server stopped answering during {task}: {len(failures)} of {len(pending)} requests failed"
                         + (f"; {fault}" if fault else f"; {detail[:200]}"))

    async def watch(self, running):
        while not self.stop.is_set():
            await asyncio.sleep(1)
        for task in running:
            task.cancel()

    async def run(self, task, items):
        out = self.run_dir / f"{task}.jsonl"
        existing = self.done(task)
        done = {record["id"] for record in existing}
        self.latencies[task] = [record["latency"] for record in existing
                                if record.get("finish") != "context" and record.get("latency")]
        counter = {"done": len(done)}
        self.report(task, counter["done"], len(items))
        gate = Gate(self.slots, self.pool)
        lock = asyncio.Lock()
        failures = []

        async def write(record):
            async with lock:
                with out.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                counter["done"] += 1
                self.report(task, counter["done"], len(items))

        async def one(item):
            if self.stop.is_set() or self.gone:
                return
            if item["meta"].get("unreachable"):
                await write(out_of_context(task, item, time.time()))
                return
            size = weight(item)
            await gate.acquire(size)
            try:
                if self.stop.is_set():
                    return
                if self.too_late(task, item):
                    self.drained = True
                    return
                started = time.time()
                try:
                    record = await self.solve(task, item, started)
                except Exception as error:
                    if not context_error(error):
                        failures.append({"id": item["id"], "error": repr(error)[:500]})
                        self.note_failure(error)
                        return
                    record = await self.retry_fitted(task, item, error, started)
                if record["finish"] != "context":
                    self.latencies[task].append(record["latency"])
                self.streak = 0
                await write(record)
            finally:
                gate.release(size)

        async def wave(batch):
            running = [asyncio.create_task(one(item)) for item in batch]
            watcher = asyncio.create_task(self.watch(running))
            results = await asyncio.gather(*running, return_exceptions=True)
            watcher.cancel()
            errors = [result for result in results if isinstance(result, Exception)]
            if errors:
                raise errors[0]

        pending = [item for item in items if item["id"] not in done]
        await wave(pending)
        if failures and not self.stop.is_set() and not self.drained and not self.gone:
            retry_ids = {failure["id"] for failure in failures}
            failures.clear()
            await asyncio.sleep(15)
            await wave([item for item in pending if item["id"] in retry_ids])
        self.check_alive(task, pending, failures)
        return failures
