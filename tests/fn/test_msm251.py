"""Verification tests for msm251.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 10, eq. 10.13 p.411, the output weight update. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm251 import mvsml_reproducing_kernel_eq_10_13


X = [[1.0, 2.0]]
Y = [[1.0]]
W = [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0]]]


def _sig(z):
    return 1.0 / (1.0 + math.exp(-z))


def _forward():
    """The hidden units and the output, by hand from eqs 10.1 to 10.3."""
    v = [_sig(1.0 * 1.0 + 0.0 * 2.0), _sig(0.0 * 1.0 + 1.0 * 2.0)]
    out = v[0] + v[1]
    return v, out


def test_the_update_adds_the_adjustment_to_the_current_weight():
    # eq 10.13: w(t+1) = w(t) + Delta w
    from morie.fn.msm250 import mvsml_ann_output_delta as delta
    step = delta(X, Y, W, eta=0.1)["delta_w"][0]
    res = mvsml_reproducing_kernel_eq_10_13(X, Y, W, eta=0.1)
    assert list(res["W"][1][0]) == pytest.approx(
        [W[1][0][j] + step[j] for j in range(2)], rel=1e-9)


def test_a_zero_learning_rate_leaves_every_weight_where_it_was():
    res = mvsml_reproducing_kernel_eq_10_13(X, Y, W, eta=0.0)
    assert list(res["W"][1][0]) == pytest.approx(W[1][0], rel=1e-12)
    assert list(res["W"][0][0]) == pytest.approx(W[0][0], rel=1e-12)


def test_one_step_of_descent_lowers_the_loss():
    from morie.fn.msm246 import mvsml_ann_gradient as grad
    before = grad(X, Y, W, eta=0.1)["loss"]
    res = mvsml_reproducing_kernel_eq_10_13(X, Y, W, eta=0.1)
    assert res["loss"] < before


def test_the_history_records_the_loss_before_the_step():
    from morie.fn.msm246 import mvsml_ann_gradient as grad
    res = mvsml_reproducing_kernel_eq_10_13(X, Y, W, eta=0.1)
    assert res["history"][0] == pytest.approx(
        grad(X, Y, W, eta=0.1)["loss"], rel=1e-12)
