from mbench import scoring


def test_choice_final_answer_formats():
    assert scoring.choice("Reasoning...\nAnswer: **C**", "C") == ("C", 1.0)
    assert scoring.choice("So the answer is (b).", "B") == ("B", 1.0)
    assert scoring.choice("It must be \\boxed{J}", "J") == ("J", 1.0)
    assert scoring.choice("I cannot decide.", "A") == (None, 0.0)


def test_math_accepts_equivalent_forms():
    assert scoring.math(r"so \boxed{\dfrac{7}{2}}", r"\frac{7}{2}")[1] == 1.0
    assert scoring.math(r"\boxed{-1/21}", r"-\frac{1}{21}")[1] == 1.0
    assert scoring.math(r"\boxed{\frac{2\sqrt{435}}{3}}", r"\frac{\sqrt{1740}}{3}")[1] == 1.0
    assert scoring.math(r"\boxed{74^{\circ}}", r"74^\circ")[1] == 1.0
    assert scoring.math(r"\boxed{420,261}", "420261")[1] == 1.0


def test_math_integer_answers_and_misses():
    assert scoring.math(r"\boxed{070}", "70")[1] == 1.0
    assert scoring.math(r"\boxed{\dfrac{487}{3}}", "204")[1] == 0.0
    assert scoring.math("no box here", "70") == (None, 0.0)


def test_math_takes_the_last_box():
    assert scoring.math(r"first \boxed{5}, then after checking \boxed{8\sqrt6}", r"8\sqrt{6}")[1] == 1.0
    assert scoring.math(r"\boxed{8\sqrt6} but finally \boxed{5}", r"8\sqrt{6}")[1] == 0.0


def test_mrcr_needs_the_prefix_and_rates_similarity():
    gold = {"answer": "abc123Roses are red, violets are blue.", "prefix": "abc123"}
    assert scoring.mrcr("abc123Roses are red, violets are blue.", gold)[1] == 1.0
    assert scoring.mrcr("\nabc123Roses are red, violets are blue.", gold)[1] == 1.0
    assert scoring.mrcr("Roses are red, violets are blue.", gold)[1] == 0.0
    assert 0.3 < scoring.mrcr("abc123Roses are red, skies are grey.", gold)[1] < 1.0


def test_graphwalks_f1_on_the_last_line():
    assert scoring.graphwalks("thinking\nFinal Answer: [a1, b2]", ["b2", "a1"])[1] == 1.0
    assert abs(scoring.graphwalks("Final Answer: [a1, c3]", ["a1", "b2"])[1] - 0.5) < 1e-9
    assert scoring.graphwalks("Final Answer: []", [])[1] == 1.0
    assert scoring.graphwalks("Final Answer: [a1]\nhope that helps", ["a1"])[1] == 0.0
    assert scoring.graphwalks("the nodes are a1 and b2", ["a1", "b2"])[1] == 0.0


def test_graphwalks_disjoint_answers_score_zero_not_one():
    assert scoring.graphwalks("Final Answer: [x9]", ["a1"])[1] == 0.0
    assert scoring.graphwalks("Final Answer: []", ["a1"])[1] == 0.0
