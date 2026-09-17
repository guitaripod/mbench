import asyncio
import random
import time

import pandas as pd
from openai import AsyncOpenAI

from . import datasets, hosts, paths, suite, swap
from .engine import request_kwargs


class Filler:
    """Deterministic long-prompt filler from MMLU-Pro question text, sized with the gpt-oss tokenizer."""

    def __init__(self):
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(datasets.fetch("tokenizer")))
        frame = pd.read_parquet(datasets.fetch("mmlupro"))
        self.paragraphs = [text.strip() for text in frame["question"].tolist() if len(text) > 200]
        self.lengths = [len(encoding.ids) for encoding in tokenizer.encode_batch(self.paragraphs)]

    def build(self, target, seed):
        order = list(range(len(self.paragraphs)))
        random.Random(seed).shuffle(order)
        picked, total = [], 0
        for index in order:
            if total >= target:
                break
            picked.append(self.paragraphs[index])
            total += self.lengths[index]
        return "\n\n".join(picked)


class SpeedRun:
    """Greedy decoding throughout, so the same prompt produces the same tokens and drafter acceptance stays comparable between runs."""

    def __init__(self, profile, spec, effort, run_id, report, abort, levels=None, context=None, host=None,
                 partial=None):
        self.profile = profile
        self.partial = partial
        self.host = host or hosts.for_profile(profile)
        self.spec = spec
        self.levels = list(levels or spec["concurrency"])
        self.context = context or profile.context
        self.effort = effort
        self.run_id = run_id
        self.report = report
        self.abort = abort
        self.client = AsyncOpenAI(base_url=(profile.base_url or paths.SWAP_URL) + "/v1", api_key="none", timeout=3600,
                                  max_retries=0)
        self.counter = 0
        self.contention = []

    def tag(self):
        """Every request opens with a unique line so no prefix-cache hit flatters prefill."""
        self.counter += 1
        return f"Request {self.run_id}-{self.counter}."

    def counters(self):
        return swap.spec_counters(self.profile.id) if self.profile.engine == "sglang" else None

    def note_contention(self):
        for process in self.host.contention(samples=1):
            if process["name"] not in [seen["name"] for seen in self.contention]:
                self.contention.append(process)

    async def stream_once(self, content, max_tokens):
        messages = [{"role": "user", "content": f"{self.tag()}\n\n{content}"}]
        started_wall = time.time()
        started = time.perf_counter()
        first = last = None
        usage = None
        finish = None
        try:
            stream = await self.client.chat.completions.create(
                model=self.profile.served, messages=messages, max_tokens=max_tokens, stream=True,
                stream_options={"include_usage": True}, **request_kwargs(self.profile, self.effort, greedy=True),
            )
            async for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                for choice in chunk.choices:
                    delta = choice.delta
                    text = (delta.content or "") + (getattr(delta, "reasoning_content", None) or "")
                    if text:
                        now = time.perf_counter()
                        first = first if first is not None else now
                        last = now
                    if choice.finish_reason:
                        finish = choice.finish_reason
        except Exception as error:
            return {"start": started_wall, "end": time.time(), "error": repr(error)[:300]}
        ended = time.perf_counter()
        completion = usage.completion_tokens if usage else 0
        window = (last - first) if first is not None and last is not None else 0
        return {
            "start": started_wall,
            "end": started_wall + (ended - started),
            "ttft_s": round(first - started, 4) if first is not None else None,
            "decode_tps": round((completion - 1) / window, 2) if window > 0 and completion > 1 else None,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": completion,
            "finish_reason": finish,
        }

    def keep(self, results):
        """Writes what has been measured so far. A phone run interrupted at answer nineteen of twenty used to lose the
        whole phase; now it loses the answer."""
        if self.partial:
            self.partial(results)

    def cool(self):
        """Lets the device come back to its cold state so the next block is not measured on heat the block before it
        made. A card returns at once."""
        found = self.host.cooldown()
        if found.get("waited_s"):
            self.report("cooling", 0, 1)
        return found

    async def sustained(self, text, spec, sampler, results, step):
        """Decodes under near-continuous load, one second between answers, the protocol the published sustained-load
        measurements of phone inference use. The first answers run at the cold clock and the rest at whatever the
        chassis can hold."""
        for index in range(spec["reps"]):
            if self.abort.is_set():
                return
            if index:
                await asyncio.sleep(spec.get("gap_s", 1))
            row = await self.stream_once(text, spec["max_tokens"])
            row.update(index=index, **sampler.window(row["start"], row["end"]))
            results["sustain"].append(row)
            step()

    def depths(self):
        context = self.context
        limit = self.spec["depth_max_tokens"] + 1024
        return [depth for depth in self.spec["depths"] if context is None or depth + limit <= context]

    async def run(self):
        prompts = suite.canonical_prompts()
        depths = self.depths()
        sustain = self.spec.get("sustain")
        total = (len(prompts) * self.spec["reps"] + len(self.levels) * self.spec["rounds"]
                 + len(depths) * self.spec["depth_reps"] + (sustain["reps"] if sustain else 0))
        progress = {"done": 0}
        results = {"single": [], "concurrency": [], "depth": [], "sustain": [], "cooldowns": [],
                   "depths_skipped": sorted(set(self.spec["depths"]) - set(depths))}

        def step():
            progress["done"] += 1
            self.report("speed", progress["done"], total)
            self.keep(results)

        sampler = self.host.sampler()
        try:
            await self.stream_once("Say hello in five words.", 64)
            await self.stream_once(prompts["code-v1"], 256)
            self.report("speed", 0, total)
            if sustain:
                results["cooldowns"].append({"before": "sustain", **self.cool()})
                await self.sustained(prompts["prose-v1"], sustain, sampler, results, step)
                results["cooldowns"].append({"before": "throughput", **self.cool()})
            for name, text in prompts.items():
                for rep in range(self.spec["reps"]):
                    if self.abort.is_set():
                        return results
                    self.note_contention()
                    before = self.counters()
                    row = await self.stream_once(text, self.spec["max_tokens"])
                    row.update(prompt=name, rep=rep, **sampler.window(row["start"], row["end"]))
                    row["accept_length"] = swap.accept_length(before, self.counters())
                    results["single"].append(row)
                    step()
            texts = list(prompts.values())
            for level in self.levels:
                for round_index in range(self.spec["rounds"]):
                    if self.abort.is_set():
                        return results
                    self.note_contention()
                    before = self.counters()
                    streams = await asyncio.gather(*(self.stream_once(texts[index % len(texts)], self.spec["max_tokens"])
                                                     for index in range(level)))
                    good = [stream for stream in streams if not stream.get("error")]
                    if good:
                        start = min(stream["start"] for stream in good)
                        end = max(stream["end"] for stream in good)
                        total_tokens = sum(stream["completion_tokens"] for stream in good)
                        results["concurrency"].append({
                            "concurrency": level, "round": round_index,
                            "aggregate_tps": round(total_tokens / (end - start), 1),
                            "streams": streams, **sampler.window(start, end),
                            "accept_length": swap.accept_length(before, self.counters()),
                        })
                    step()
            filler = Filler()
            for depth in depths:
                for rep in range(self.spec["depth_reps"]):
                    if self.abort.is_set():
                        return results
                    self.note_contention()
                    content = filler.build(depth, f"{depth}-{rep}") + (
                        "\n\nSummarize the main subjects covered by the questions above as a detailed bulleted list.")
                    before = self.counters()
                    row = await self.stream_once(content, self.spec["depth_max_tokens"])
                    row.update(depth=depth, rep=rep, **sampler.window(row["start"], row["end"]))
                    row["accept_length"] = swap.accept_length(before, self.counters())
                    if row.get("ttft_s") and row.get("prompt_tokens"):
                        row["prefill_tps"] = round(row["prompt_tokens"] / row["ttft_s"], 1)
                    results["depth"].append(row)
                    step()
        finally:
            results["telemetry"] = sampler.timeline()
            sampler.close()
            results["contention"] = self.contention
        return results
