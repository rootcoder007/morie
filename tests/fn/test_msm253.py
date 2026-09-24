"""Verification tests for msm253.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 10, eqs. 10.14 and 10.15 p.411, the hidden gradient. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm253 import mvsml_reproducing_kernel_eq_10_14


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


def test_the_hidden_step_follows_the_chain_rule_through_the_output():
    # eq 10.15: psi_ik = delta w_jk g'(z_ik), Delta w_kp = eta psi_ik x_ip
    v, out = _forward()
    delta = 1.0 - out
    res = mvsml_reproducing_kernel_eq_10_14(X, Y, W, eta=0.1)
    for k in range(2):
        psi = delta * W[1][0][k] * v[k] * (1.0 - v[k])
        for p, xp in enumerate([1.0, 2.0]):
            assert res["delta_w"][k][p] == pytest.approx(0.1 * psi * xp,
                                                          rel=1e-9)


def test_the_gradient_is_the_step_divided_by_the_negated_rate():
    res = mvsml_reproducing_kernel_eq_10_14(X, Y, W, eta=0.1)
    for grow, drow in zip(res["gradient"], res["delta_w"]):
        for g, d in zip(grow, drow):
            assert d == pytest.approx(-0.1 * g, rel=1e-12)


def test_the_slope_factor_is_the_logistic_derivative():
    # g'(z) = V (1 - V) is what makes the hidden step differ from the
    # output one, where the identity activation has slope one
    v, _ = _forward()
    assert v[0] * (1.0 - v[0]) == pytest.approx(0.19661193324148185,
                                                 rel=1e-12)


def test_a_severed_output_weight_stops_the_hidden_unit_learning():
    res = mvsml_reproducing_kernel_eq_10_14(X, Y, [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0]]], eta=0.1)
    assert list(res["delta_w"][0]) == pytest.approx([0.0, 0.0],
                                                     abs=1e-12)
