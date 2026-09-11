import hashlib

from mbench import lmx
from mbench.engine import injection
from mbench.profiles import Profile


def profile(**fields):
    return Profile(**{"id": "m", "name": "m", "engine": "sglang", "thinking": "qwen", "context": 524288, **fields})


def test_payload_keeps_api_fields_and_adds_verification_evidence():
    run = {"hfId": "org/m", "hardware": {"hwClass": "DISCRETE_GPU"}, "engineName": "sglang", "quantization": "FP8",
           "tokSOut": 150.0, "prompt": "p", "outputText": "o", "samples": [{"iteration": 1}], "agentFeedback": {"x": 1}}
    body = lmx.payload(run, profile())
    assert "agentFeedback" not in body and "prompt" not in body
    assert body["contextLength"] == 524288
    assert body["promptSha256"] == hashlib.sha256(b"p").hexdigest()
    assert body["outputSample"] == "o"
    assert body["engineTimingsRaw"]["samples"] == [{"iteration": 1}]


def test_long_outputs_are_sampled_head_and_tail():
    body = lmx.payload({"prompt": "p", "outputText": "a" * 3500 + "b" * 1500}, profile())
    assert len(body["outputSample"]) <= 4000
    assert body["outputSample"].startswith("a" * 2996) and body["outputSample"].endswith("b" * 1000)


def test_spec_counts_add_up_to_the_output_tokens():
    counts = lmx.spec_counts(profile(spec={"tokens_per_step": 7}), 512, 4.4)
    assert abs(counts["specAcceptedTokens"] + counts["specDraftTokens"] / 7 - 512) <= 1
    assert lmx.spec_counts(profile(), 512, 4.4) == {}


def test_injection_flattens_the_thinking_setting():
    assert injection(profile(), "xhigh") == {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}}
    assert injection(profile(thinking="openai"), "high") == {"reasoning_effort": "high"}
    assert injection(profile(thinking="none"), "high") == {}
