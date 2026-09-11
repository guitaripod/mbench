import asyncio
import hashlib
import json
import time

from openai import AsyncOpenAI

from . import paths, scoring

QWEN_EFFORTS = {"high": "xhigh"}


def supports_seed(profile):
    """SGLang's FlashInfer sampler asserts on seeded top-k/top-p requests and would take the server down, so only llama.cpp gets seeds."""
    return profile.engine == "llama.cpp"


def seed_for(item_id):
    return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) % 2_147_483_647


def request_kwargs(profile, effort, *, greedy=False, seed=None):
    """How each chat template takes a thinking level; sampling stays on the server's model defaults unless greedy."""
    kwargs, extra = {}, {}
    if profile.thinking == "openai":
        kwargs["reasoning_effort"] = effort
    elif profile.thinking == "qwen":
        extra["chat_template_kwargs"] = {"enable_thinking": True, "reasoning_effort": QWEN_EFFORTS.get(effort, effort)}
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


def tool_calls(message):
    calls, malformed = [], 0
    for call in message.tool_calls or []:
        try:
            calls.append([call.function.name, json.loads(call.function.arguments or "{}")])
        except json.JSONDecodeError:
            malformed += 1
    return calls, malformed


def build_record(task, item, response, started):
    choice = response.choices[0]
    message = choice.message
    content = message.content or ""
    record = {
        "id": item["id"],
        "task": task,
        "sample": item["sample"],
        "gold": item["gold"],
        "meta": item["meta"],
        "finish": choice.finish_reason,
        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
        "completion_tokens": response.usage.completion_tokens if response.usage else None,
        "latency": round(time.time() - started, 2),
        "content": content,
        "reasoning": getattr(message, "reasoning_content", None) or "",
    }
    if task == "tools":
        calls, malformed = tool_calls(message)
        correct = malformed == 0 and scoring.tools(item["gold"], calls)
        record.update(prediction=calls, malformed=malformed, score=float(correct))
    elif task == "lcb":
        record.update(prediction=None, score=None)
    else:
        prediction, score = scoring.TEXT_SCORERS[task](content, item["gold"])
        record.update(prediction=prediction, score=float(score))
    return record


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

    async def call(self, item):
        kwargs = request_kwargs(self.profile, self.effort, seed=seed_for(item["id"]))
        if item.get("tools"):
            kwargs.update(tools=item["tools"], tool_choice="auto")
        return await self.client.chat.completions.create(
            model=self.profile.id, messages=item["messages"], max_tokens=item["max_tokens"], **kwargs
        )

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
                try:
                    response = await self.call(item)
                except Exception as error:
                    failures.append({"id": item["id"], "error": repr(error)[:500]})
                    return
                record = build_record(task, item, response, started)
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
