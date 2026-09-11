import asyncio
import json
import threading
import time

import pytest

from mbench import engine
from mbench.engine import Gate, QualityRunner
from mbench.profiles import Profile


def profile(thinking="qwen", efforts=("low", "medium", "xhigh")):
    return Profile(id="m", name="m", engine="sglang", thinking=thinking, context=None, efforts=list(efforts))


def test_the_gate_holds_slots_and_shared_context_and_keeps_order():
    async def scenario():
        gate = Gate(slots=3, pool=100)
        order = []

        async def job(name, size, hold):
            await gate.acquire(size)
            order.append(("start", name))
            await asyncio.sleep(hold)
            gate.release(size)
            order.append(("end", name))

        await asyncio.gather(job("a", 60, 0.05), job("b", 60, 0.01), job("c", 10, 0.01))
        return order

    order = asyncio.run(scenario())
    assert order[0] == ("start", "a")
    assert order.index(("start", "b")) > order.index(("end", "a"))
    assert order.index(("start", "c")) > order.index(("start", "b"))


def test_an_item_bigger_than_the_pool_runs_alone():
    async def scenario():
        gate = Gate(slots=4, pool=100)
        await gate.acquire(500)
        waiting = asyncio.create_task(gate.acquire(1))
        await asyncio.sleep(0.01)
        blocked = not waiting.done()
        gate.release(500)
        await asyncio.wait_for(waiting, 1)
        return blocked

    assert asyncio.run(scenario())


def test_a_cancelled_waiter_does_not_hold_the_gate():
    async def scenario():
        gate = Gate(slots=1)
        await gate.acquire(1)
        waiter = asyncio.create_task(gate.acquire(1))
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.sleep(0)
        gate.release(1)
        await asyncio.wait_for(gate.acquire(1), 1)
        return gate.running

    assert asyncio.run(scenario()) == 1


def test_item_weight_uses_counted_tokens_and_caps_the_answer():
    assert engine.weight({"meta": {"tokens": 100000}, "messages": [], "max_tokens": 65536}) == 100000 + engine.ANSWER_RESERVE
    assert engine.weight({"meta": {}, "messages": [{"content": "x" * 3000}], "max_tokens": 1000}) == 1000 + 512 + 1000


def items(count, max_tokens=400):
    return [{"id": f"supergpqa-{index}", "task": "supergpqa", "messages": [{"role": "user", "content": "q"}], "gold": "A",
             "meta": {}, "max_tokens": max_tokens, "sample": 0, "tools": None} for index in range(count)]


def runner(tmp_path, deadline=None, halt=None, solve=None):
    instance = QualityRunner(profile(), tmp_path, "medium", lambda *args: None, halt or threading.Event(), slots=2,
                             deadline=deadline)

    async def answer(task, item, started):
        await asyncio.sleep(0.01)
        return {**engine.base_record(task, item, "stop", started), "prediction": "A", "score": 1.0}

    instance.solve = solve or answer
    return instance


def test_items_that_could_not_finish_before_the_deadline_are_not_started(tmp_path):
    run = runner(tmp_path, deadline=time.time() + 5)
    asyncio.run(run.run("supergpqa", items(3)))
    assert run.drained and not (tmp_path / "supergpqa.jsonl").exists()


def test_the_pace_so_far_decides_what_still_fits(tmp_path):
    (tmp_path / "supergpqa.jsonl").write_text("".join(
        json.dumps({"id": f"old-{index}", "latency": 1.0, "finish": "stop"}) + "\n" for index in range(3)))
    run = runner(tmp_path, deadline=time.time() + 5)
    asyncio.run(run.run("supergpqa", items(3)))
    assert not run.drained
    assert len((tmp_path / "supergpqa.jsonl").read_text().splitlines()) == 6


def test_a_halt_cancels_requests_in_flight(tmp_path):
    halt = threading.Event()

    async def slow(task, item, started):
        halt.set()
        await asyncio.sleep(30)

    run = runner(tmp_path, halt=halt, solve=slow)
    started = time.time()
    asyncio.run(run.run("supergpqa", items(2)))
    assert time.time() - started < 5
    assert not (tmp_path / "supergpqa.jsonl").exists()


def test_thinking_can_be_switched_off_where_the_template_allows_it():
    qwen = profile()
    assert engine.resolve_effort(qwen, "none") == "none"
    assert engine.request_kwargs(qwen, "none")["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    with pytest.raises(ValueError, match="can't switch thinking off"):
        engine.resolve_effort(profile("openai", ("low", "medium", "high")), "none")
    declared = profile("openai", ("none", "low", "medium"))
    assert engine.resolve_effort(declared, "none") == "none"
    assert engine.request_kwargs(declared, "none")["reasoning_effort"] == "none"
