import pytest

from mbench.engine import request_kwargs, resolve_effort
from mbench.profiles import Profile


def profile(thinking="qwen", engine="sglang", efforts=("low", "medium", "xhigh")):
    return Profile(id="m", name="m", engine=engine, thinking=thinking, context=None, efforts=list(efforts))


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
