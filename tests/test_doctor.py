import asyncio
import json
from types import SimpleNamespace

from mbench import doctor
from mbench.profiles import Profile

PROFILE = Profile(id="m", name="m", engine="sglang", thinking="qwen", context=None, efforts=["medium"])
ROOMY = {"slots": 4, "context": 262144, "pool": 262144}


def reply(content="", reasoning="", tool_calls=None):
    message = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=16000, completion_tokens=10))


def weather_call():
    return [SimpleNamespace(id="c1", function=SimpleNamespace(name="get_weather",
                                                              arguments=json.dumps({"city": "Helsinki", "unit": "celsius"})))]


class Server:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **request):
        self.seen.append(request)
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def examine(monkeypatch, replies, capacity=ROOMY):
    server = Server(replies)
    monkeypatch.setattr(doctor, "AsyncOpenAI", lambda **kwargs: server)
    monkeypatch.setattr(doctor.datasets, "mrcr", lambda spec, context: [
        {"messages": [{"role": "user", "content": "long"}], "max_tokens": 1024, "meta": {"tokens": 16000}}])
    checks = asyncio.run(doctor.run(PROFILE, "medium", capacity["context"], capacity))
    return {check["check"]: check for check in checks}, server


def test_a_healthy_server_passes_everything(monkeypatch):
    checks, server = examine(monkeypatch, [reply("144", "12 times 12"), reply(reasoning="need the weather",
                                                                               tool_calls=weather_call()),
                                           reply("It's 14 °C and lightly raining."), reply("abc")])
    assert {name: check["status"] for name, check in checks.items()} == {
        "capacity": "ok", "answer": "ok", "tool call": "ok", "tool result": "ok", "long prompt": "ok"}
    history = server.seen[2]["messages"]
    assert history[2]["reasoning_content"] == "need the weather"
    assert history[3]["role"] == "tool" and json.loads(history[3]["content"])["temperature"] == 14


def test_parsers_missing_on_the_server_fail_the_run(monkeypatch):
    checks, _ = examine(monkeypatch, [reply("<think>hmm</think>144"), reply('<tool_call>{"name": "get_weather"}</tool_call>'),
                                      reply("abc")])
    assert checks["answer"]["status"] == "fail" and "reasoning parser" in checks["answer"]["detail"]
    assert checks["tool call"]["status"] == "fail" and "tool-call parser" in checks["tool call"]["detail"]
    assert checks["tool result"]["status"] == "warn"
    assert len(doctor.failures(list(checks.values()))) == 2


def test_a_small_context_per_request_is_a_warning(monkeypatch):
    checks, _ = examine(monkeypatch, [reply("144", "r"), reply(tool_calls=weather_call()), reply("14 °C"), reply("abc")],
                        {"slots": 4, "context": 65536, "pool": 65536})
    assert checks["capacity"]["status"] == "warn" and "64k" in checks["capacity"]["detail"]


def test_a_server_that_refuses_tool_results_fails(monkeypatch):
    checks, _ = examine(monkeypatch, [reply("144", "r"), reply(tool_calls=weather_call()), RuntimeError("400 bad role"),
                                      reply("abc")])
    assert checks["tool result"]["status"] == "fail"


def test_a_changed_stack_is_a_warning_not_a_failure():
    check = doctor.check_stack(["SGLang aaaa → SGLang bbbb"])
    assert check["status"] == "warn" and "SGLang aaaa → SGLang bbbb" in check["detail"]
    assert doctor.failures([check]) == []


def test_a_pool_that_holds_one_question_leaves_the_other_slots_idle():
    check = doctor.check_capacity(16384, {"slots": 4, "context": 16384, "pool": 16384})
    assert check["status"] == "warn" and "questions run 1 at a time" in check["detail"]
    assert "3 of the 4 slots sit idle" in check["detail"] and "will score zero" in check["detail"]
    roomy = doctor.check_capacity(262144, ROOMY)
    assert roomy["status"] == "ok" and "questions run 4 at a time" in roomy["detail"]


def test_a_phone_is_not_told_how_its_pool_would_pace_questions():
    check = doctor.check_capacity(131072, {"slots": 4, "context": 131072, "pool": 16384}, phone=True)
    assert check["status"] == "ok" and "questions" not in check["detail"]


def test_a_model_that_skips_reasoning_on_an_easy_question_is_asked_a_harder_one(monkeypatch):
    rest = [reply(tool_calls=weather_call()), reply("14 °C"), reply("abc")]
    checks, server = examine(monkeypatch, [reply("144"), reply("", "digits of n: a+b+c=10"), *rest])
    assert checks["answer"]["status"] == "ok" and "none for a question this easy" in checks["answer"]["detail"]
    assert server.seen[1]["messages"][0]["content"] == doctor.HARDER_QUESTION
    checks, _ = examine(monkeypatch, [reply("144"), reply("63"), *rest])
    assert checks["answer"]["status"] == "warn" and "even to a harder question" in checks["answer"]["detail"]


def test_the_fingerprint_keeps_a_greedy_answer(monkeypatch):
    server = Server([reply("2 3 5 7 11 13 17 19", "primes: ")])
    monkeypatch.setattr(doctor, "AsyncOpenAI", lambda **kwargs: server)
    probe = asyncio.run(doctor.fingerprint(PROFILE, "medium"))
    assert probe == {"prompt": doctor.FINGERPRINT_PROMPT, "answer": "primes: 2 3 5 7 11 13 17 19"}
    assert server.seen[0]["temperature"] == 0
