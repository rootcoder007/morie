"""Verification tests for msm246.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 10, eqs. 10.10 and 10.11 p.410, the gradient step. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm246 import mvsml_reproducing_kernel_eq_10_10


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


def test_the_loss_is_half_the_squared_error_of_the_forward_pass():
    _, out = _forward()
    res = mvsml_reproducing_kernel_eq_10_10(X, Y, W)
    assert res["loss"] == pytest.approx(0.5 * (1.0 - out) ** 2, rel=1e-12)


def test_the_weight_change_is_the_negated_scaled_gradient():
    # eq 10.10: Delta w = -eta dE/dw
    res = mvsml_reproducing_kernel_eq_10_10(X, Y, W, eta=0.1)
    for gl, dl in zip(res["gradients"], res["weight_changes"]):
        for grow, drow in zip(gl, dl):
            for g, d in zip(grow, drow):
                assert d == pytest.approx(-0.1 * g, rel=1e-12)


def test_a_larger_learning_rate_scales_the_step_proportionally():
    small = mvsml_reproducing_kernel_eq_10_10(X, Y, W, eta=0.1)["weight_changes"][1][0][0]
    large = mvsml_reproducing_kernel_eq_10_10(X, Y, W, eta=0.4)["weight_changes"][1][0][0]
    assert large == pytest.approx(4.0 * small, rel=1e-12)


def test_the_output_gradient_follows_the_chain_rule_of_equation_10_11():
    # dE/dw = -(y - y-hat) V, with an identity output activation
    v, out = _forward()
    res = mvsml_reproducing_kernel_eq_10_10(X, Y, W, eta=0.1)
    expected = [-(1.0 - out) * vj for vj in v]
    assert list(res["gradients"][1][0]) == pytest.approx(expected,
                                                          rel=1e-12)
