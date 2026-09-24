"""Verification tests for grn021.

Geron (2023), *Hands-On Machine Learning with Scikit-Learn, Keras and
TensorFlow*, 3rd edition, ch 4, eq. 4.20, the softmax function. Expected values are recomputed in
the test body and the docstring's own worked value is asserted too.
"""

import math

import pytest

from morie.fn.grn021 import geron_ch4_softmax_function


def test_the_score_is_exponentiated_and_divided_by_the_total():
    res = geron_ch4_softmax_function([0.0, 1.0, 2.0], k=2, K=3)
    total = 1.0 + math.exp(1.0) + math.exp(2.0)
    assert res["probability"] == pytest.approx(math.exp(2.0) / total,
                                                rel=1e-12)
    assert round(res["probability"], 6) == pytest.approx(0.665241,
                                                          abs=1e-6)


def test_the_probabilities_sum_to_one():
    res = geron_ch4_softmax_function([0.0, 1.0, 2.0], k=2, K=3)
    assert round(sum(res["probabilities"]), 12) == pytest.approx(
        1.0, abs=1e-12)


def test_adding_a_constant_to_every_score_changes_nothing():
    a = geron_ch4_softmax_function([0.0, 1.0, 2.0], k=2, K=3)["probability"]
    b = geron_ch4_softmax_function([100.0, 101.0, 102.0], k=2, K=3)["probability"]
    assert a == pytest.approx(b, rel=1e-12)
    assert round(b, 6) == pytest.approx(0.665241, abs=1e-6)


def test_equal_scores_split_the_mass_evenly():
    res = geron_ch4_softmax_function([1.0, 1.0, 1.0], k=0, K=3)
    assert res["probability"] == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_the_largest_score_takes_the_largest_share():
    res = geron_ch4_softmax_function([0.0, 1.0, 2.0], k=2, K=3)
    probs = list(res["probabilities"])
    assert probs[2] == max(probs)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grn021 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
