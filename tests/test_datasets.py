import pytest

from mbench import datasets, suite


def cached(name):
    try:
        datasets.fetch(name)
        return True
    except Exception:
        return False


needs_data = pytest.mark.skipif(not (cached("mmlupro") and cached("lcb") and cached("aime") and cached("tokenizer")),
                                reason="pinned datasets are not cached on this machine")


@needs_data
def test_full_suite_sizes_and_order_are_stable():
    for task, size in (("mmlupro", 420), ("aime", 120), ("lcb", 101), ("tools", 60)):
        first = [entry["id"] for entry in datasets.build(task, suite.SUITES["full"][task])]
        second = [entry["id"] for entry in datasets.build(task, suite.SUITES["full"][task])]
        assert first == second
        assert len(first) == size


@needs_data
def test_needle_prompts_are_identical_between_builds():
    spec = {"depths": (16000,), "per_depth": 2, "max_tokens": 4096}
    first = datasets.build("niah", spec)
    second = datasets.build("niah", spec)
    assert [entry["messages"] for entry in first] == [entry["messages"] for entry in second]


@needs_data
def test_needle_depths_respect_the_context_window():
    spec = {"depths": (16000, 120000), "per_depth": 1, "max_tokens": 8192}
    items = datasets.build("niah", spec, context=65536)
    assert {entry["meta"]["depth"] for entry in items} == {16000}
    assert datasets.build("niah", spec, context=None)[0]["messages"] == items[0]["messages"]
