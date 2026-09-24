"""Verification tests for msm250.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 10, eq. 10.12 p.411, the output delta rule. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm250 import mvsml_reproducing_kernel_eq_10_12


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


def test_the_output_delta_is_the_error_times_the_activation_slope():
    # eq 10.12: Delta w = eta delta V, delta = (y - y-hat) g'(z)
    v, out = _forward()
    res = mvsml_reproducing_kernel_eq_10_12(X, Y, W, eta=0.1)
    delta = 1.0 - out
    assert list(res["delta_w"][0]) == pytest.approx(
        [0.1 * delta * vj for vj in v], rel=1e-12)


def test_the_gradient_is_the_delta_step_turned_around():
    res = mvsml_reproducing_kernel_eq_10_12(X, Y, W, eta=0.1)
    for d, g in zip(res["delta_w"][0], res["gradient"][0]):
        assert d == pytest.approx(-0.1 * g, rel=1e-12)


def test_a_perfect_prediction_asks_for_no_change_at_all():
    _, out = _forward()
    res = mvsml_reproducing_kernel_eq_10_12(X, [[out]], W, eta=0.1)
    assert list(res["delta_w"][0]) == pytest.approx([0.0, 0.0],
                                                     abs=1e-12)


def test_the_step_points_the_output_towards_the_target():
    # the prediction overshoots here, so the weights must come down
    res = mvsml_reproducing_kernel_eq_10_12(X, Y, W, eta=0.1)
    assert all(d < 0 for d in res["delta_w"][0])
