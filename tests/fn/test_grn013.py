"""Verification tests for grn013.

Geron (2023), *Hands-On Machine Learning with Scikit-Learn, Keras and
TensorFlow*, 3rd edition, ch 4, eq. 4.12, the elastic net cost. Expected values are recomputed in
the test body and the docstring's own worked value is asserted too.
"""

import math

import pytest

from morie.fn.grn013 import geron_ch4_elastic_net_cost_function


X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
Y = [1.0, 2.0, 3.0]
THETA = [0.0, 1.0]


def test_the_cost_adds_both_penalty_arms_to_a_perfect_fit():
    # the 3rd edition writes the lasso arm as 2 alpha sum |theta| and
    # the ridge arm as (alpha / m) sum theta^2
    res = geron_ch4_elastic_net_cost_function(X, Y, THETA, alpha=1.0, r=0.5)
    l1 = 0.5 * 2.0 * 1.0 * 1.0
    l2 = 0.5 * (1.0 / 3.0) * 1.0
    assert res["cost"] == pytest.approx(0.0 + l1 + l2, rel=1e-12)
    assert round(res["cost"], 10) == pytest.approx(1.1666666667,
                                                    abs=1e-10)
    assert round(res["l2_penalty"], 10) == pytest.approx(0.1666666667,
                                                          abs=1e-10)


def test_the_ridge_arm_carries_a_real_one_over_m():
    X6 = X + [[1.0, 4.0], [1.0, 5.0], [1.0, 6.0]]
    y6 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    res = geron_ch4_elastic_net_cost_function(X6, y6, THETA, alpha=1.0, r=0.5)
    assert round(res["l2_penalty"], 10) == pytest.approx(0.0833333333,
                                                          abs=1e-10)
    assert res["l2_penalty"] == pytest.approx(0.5 * (1.0 / 6.0) * 1.0,
                                               rel=1e-12)


def test_a_mixing_ratio_of_one_leaves_pure_lasso():
    res = geron_ch4_elastic_net_cost_function(X, Y, THETA, alpha=1.0, r=1.0)
    assert res["l2_penalty"] == pytest.approx(0.0, abs=1e-15)


def test_a_mixing_ratio_of_zero_leaves_pure_ridge():
    res = geron_ch4_elastic_net_cost_function(X, Y, THETA, alpha=1.0, r=0.0)
    assert res["l1_penalty"] == pytest.approx(0.0, abs=1e-15)


def test_the_intercept_is_left_out_of_the_penalty_by_default():
    with_bias = geron_ch4_elastic_net_cost_function(X, Y, [5.0, 1.0], alpha=1.0, r=0.5)
    without = geron_ch4_elastic_net_cost_function(X, Y, [0.0, 1.0], alpha=1.0, r=0.5)
    assert with_bias["l1_penalty"] == pytest.approx(without["l1_penalty"],
                                                     rel=1e-12)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grn013 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
