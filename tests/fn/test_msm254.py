"""Verification tests for msm254.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 10, eq. 10.16 p.412, the hidden delta rule. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm254 import mvsml_reproducing_kernel_eq_10_16


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


def test_the_hidden_delta_rule_agrees_with_the_expanded_gradient():
    # eq 10.16 is eq 10.14 written with psi collected into one symbol
    from morie.fn.msm253 import mvsml_ann_hidden_gradient as expanded
    a = mvsml_reproducing_kernel_eq_10_16(X, Y, W, eta=0.1)["delta_w"]
    b = expanded(X, Y, W, eta=0.1)["delta_w"]
    for ra, rb in zip(a, b):
        assert list(ra) == pytest.approx(list(rb), rel=1e-12)


def test_the_step_is_the_rate_times_the_signal_times_the_input():
    v, out = _forward()
    delta = 1.0 - out
    res = mvsml_reproducing_kernel_eq_10_16(X, Y, W, eta=0.1)
    psi = delta * W[1][0][0] * v[0] * (1.0 - v[0])
    assert res["delta_w"][0][1] == pytest.approx(0.1 * psi * 2.0,
                                                  rel=1e-9)


def test_a_zero_rate_produces_no_hidden_step():
    res = mvsml_reproducing_kernel_eq_10_16(X, Y, W, eta=0.0)
    for row in res["delta_w"]:
        assert list(row) == pytest.approx([0.0, 0.0], abs=1e-15)
