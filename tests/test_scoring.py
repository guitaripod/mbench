from mbench import scoring


def test_mmlupro_final_answer_formats():
    assert scoring.mmlupro("Reasoning...\nAnswer: **C**", "C") == ("C", 1.0)
    assert scoring.mmlupro("So the answer is (b).", "B") == ("B", 1.0)
    assert scoring.mmlupro("It must be \\boxed{D}", "D") == ("D", 1.0)
    assert scoring.mmlupro("I cannot decide.", "A") == (None, 0.0)


def test_aime_takes_the_value_after_an_equation_chain():
    content = r"\boxed{\; \mathbb{E}[R]=n+1+\mathbb{E}[I]=27+1+176=204 \; }"
    assert scoring.aime(content, "204") == ("204", 1.0)


def test_aime_rejects_fractions_and_accepts_leading_zeros():
    assert scoring.aime(r"\boxed{\dfrac{487}{3}}", "204")[1] == 0.0
    assert scoring.aime(r"\boxed{070}", "70") == ("70", 1.0)
    assert scoring.aime("no box here", "70") == (None, 0.0)


def test_niah_accepts_non_breaking_hyphens():
    gold = {"scarlet-otter": "1133098", "copper-badger": "6675485"}
    found, score = scoring.niah("scarlet‑otter: 1133098\ncopper‑badger: 6675485", gold)
    assert score == 1.0
    assert found == gold


def test_niah_partial_credit():
    gold = {"iron-ember": "5368025", "rapid-prism": "3135960"}
    assert scoring.niah("iron-ember: 5368025\nrapid-prism: 1111111", gold)[1] == 0.5


def test_tools_parallel_calls_any_order():
    expected = [["get_weather", {"city": "Tokyo"}], ["get_weather", {"city": "Paris"}]]
    calls = [["get_weather", {"city": "paris"}], ["get_weather", {"city": "Tokyo", "unit": "celsius"}]]
    assert scoring.tools(expected, calls)
    assert not scoring.tools(expected, calls[:1])


def test_tools_no_call_expected():
    assert scoring.tools([], [])
    assert not scoring.tools([], [["get_weather", {"city": "Oslo"}]])


def test_tools_numbers_and_lists_normalise():
    assert scoring.tools([["convert_currency", {"amount": 250}]], [["convert_currency", {"amount": 250.0}]])
    assert scoring.tools([["git_commit", {"files": ["b", "a"]}]], [["git_commit", {"files": ["A", "B"]}]])
