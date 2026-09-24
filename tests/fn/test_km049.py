"""Verification tests for km049.

Kamath, Keenan, Somers and Sorenson (2024), eq. 3.4, top-one prompt
accuracy. ``P_LM(x, t)`` returns a label distribution, so the tests
check the argmax rule and the validation that distribution must pass.
"""

import pytest

from morie.fn.km049 import kamath_ch3_top1_prompt_metric


def test_top_one_accuracy_counts_argmax_agreements():
    # Eq 3.4: A(t) = (1/|R|) sum 1[y = argmax_y' P_LM(y'|x, t)]
    R = [("a", "pos"), ("b", "neg")]
    P = lambda x, t: {"pos": 0.9, "neg": 0.1}
    res = kamath_ch3_top1_prompt_metric(R, "T1", P)
    # the model always prefers "pos", so it is right on exactly one
    assert res["estimate"] == pytest.approx(0.5, rel=1e-12)
    assert res["n_correct"] == 1
    assert list(res["correct"]) == [1, 0]


def test_a_model_that_always_names_the_gold_label_scores_one():
    gold = {"a": "pos", "b": "neg"}
    P = lambda x, t: {k: (1.0 if k == gold[x] else 0.0) for k in ("pos", "neg")}
    res = kamath_ch3_top1_prompt_metric([("a", "pos"), ("b", "neg")], "T", P)
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_a_model_that_always_names_the_other_label_scores_zero():
    P = lambda x, t: {"pos": 0.0, "neg": 1.0}
    res = kamath_ch3_top1_prompt_metric([("a", "pos"), ("b", "pos")], "T", P)
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_a_distribution_that_does_not_sum_to_one_is_refused():
    with pytest.raises(ValueError):
        kamath_ch3_top1_prompt_metric([("a", "pos")], "T", lambda x, t: {"pos": 0.4, "neg": 0.4})


def test_a_gold_label_absent_from_the_distribution_is_refused():
    with pytest.raises(ValueError):
        kamath_ch3_top1_prompt_metric([("a", "other")], "T", lambda x, t: {"pos": 0.5, "neg": 0.5})


def test_accuracy_over_no_examples_is_refused_rather_than_called_zero():
    with pytest.raises(ValueError):
        kamath_ch3_top1_prompt_metric([], "T", lambda x, t: {"pos": 1.0})


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.km049 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
