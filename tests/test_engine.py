from mbench.engine import request_kwargs
from mbench.profiles import Profile


def profile(thinking, engine="sglang"):
    return Profile(id="m", name="m", engine=engine, thinking=thinking, context=None)


def test_qwen_template_gets_xhigh_for_high():
    kwargs = request_kwargs(profile("qwen"), "high")
    assert kwargs["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "xhigh"


def test_openai_style_passes_effort_and_sglang_gets_no_seed():
    kwargs = request_kwargs(profile("openai"), "low", seed=7)
    assert kwargs == {"reasoning_effort": "low"}
    assert request_kwargs(profile("none", "llama.cpp"), "medium", seed=7) == {"seed": 7}
