import pytest

from mbench import datasets, suite

REQUIRED = ("supergpqa", "aime2026", "hmmt2026", "lcb", "mrcr0", "mrcr1", "graphwalks", "tokenizer")
needs_data = pytest.mark.skipif(not all(datasets.cached_path(name).exists() for name in REQUIRED),
                                reason="pinned datasets are not cached on this machine")


def test_length_bins():
    bins = (16384, 32768, 65536, 131072)
    assert datasets.bin_of(12000, bins) == 16384
    assert datasets.bin_of(5000, bins) is None
    assert datasets.bin_of(32768, bins) == 32768
    assert datasets.bin_of(32769, bins) == 65536
    assert datasets.bin_of(140000, bins) is None
    assert datasets.bin_of(32150, (8192, 16384, 32768), 1.05) == 32768


def test_answer_budget_fits_the_window():
    assert datasets.answer_budget(None, 100000, 16384) == 16384
    assert datasets.answer_budget(131072, 60000, 16384) == 16384
    assert datasets.answer_budget(131072, 120000, 16384) == 131072 - 120000 - datasets.TEMPLATE_TOKENS
    assert datasets.answer_budget(131072, 130000, 16384) is None


@needs_data
def test_full_suite_sizes_and_order_are_stable():
    sizes = {"supergpqa": 521, "math": 126, "lcb": 101, "tools": 114, "mrcr": 60, "graphwalks": 48}
    for task, size in sizes.items():
        first = [entry["id"] for entry in datasets.build(task, suite.SUITES["full"][task])]
        second = [entry["id"] for entry in datasets.build(task, suite.SUITES["full"][task])]
        assert first == second, task
        assert len(first) == size == len(set(first)), task


@needs_data
def test_quick_items_are_a_subset_of_full_ones_for_long_context():
    for task in ("mrcr", "graphwalks"):
        quick = {entry["id"] for entry in datasets.build(task, suite.SUITES["quick"][task])}
        full = {entry["id"] for entry in datasets.build(task, suite.SUITES["full"][task])}
        assert len(quick) == {"mrcr": 20, "graphwalks": 16}[task]
        assert quick <= full, task


@needs_data
def test_long_prompts_past_the_window_become_unreachable_items():
    spec = suite.SUITES["full"]["mrcr"]
    open_window = datasets.build("mrcr", spec)
    small = datasets.build("mrcr", spec, context=65536)
    assert [entry["id"] for entry in small] == [entry["id"] for entry in open_window]
    assert not any(entry["meta"].get("unreachable") for entry in open_window)
    unreachable = {entry["meta"]["bin"] for entry in small if entry["meta"].get("unreachable")}
    assert unreachable >= {131072} and 16384 not in unreachable
    for entry in small:
        if not entry["meta"].get("unreachable"):
            assert entry["meta"]["tokens"] + entry["max_tokens"] <= 65536


@needs_data
def test_supergpqa_follows_the_full_sets_discipline_mix():
    items = datasets.build("supergpqa", suite.SUITES["full"]["supergpqa"])
    disciplines = [entry["meta"]["discipline"] for entry in items]
    assert disciplines.count("Science") > disciplines.count("Engineering") > disciplines.count("Medicine")
    assert len(set(disciplines)) == 13
    assert all(entry["gold"] in datasets.LETTERS for entry in items)


@needs_data
def test_math_has_both_competitions_with_symbolic_answers():
    items = datasets.build("math", {"samples": 1, "max_tokens": 1})
    assert {entry["meta"]["competition"] for entry in items} == {"aime", "hmmt"}
    assert any("\\frac" in entry["gold"] for entry in items)
