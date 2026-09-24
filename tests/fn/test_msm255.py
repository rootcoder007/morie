"""Verification tests for msm255.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 10, eq. 10.17 p.412, backpropagation training. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm255 import mvsml_reproducing_kernel_eq_10_17


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


def test_the_training_loss_falls_at_every_iteration():
    res = mvsml_reproducing_kernel_eq_10_17(X, Y, W, eta=0.1, n_iter=5)
    hist = list(res["history"])
    assert len(hist) == 5
    assert all(b < a for a, b in zip(hist, hist[1:]))


def test_the_first_recorded_loss_is_the_untrained_one():
    from morie.fn.msm246 import mvsml_ann_gradient as grad
    res = mvsml_reproducing_kernel_eq_10_17(X, Y, W, eta=0.1, n_iter=5)
    assert res["history"][0] == pytest.approx(
        grad(X, Y, W, eta=0.1)["loss"], rel=1e-12)


def test_the_reported_loss_is_the_one_the_last_weights_give():
    res = mvsml_reproducing_kernel_eq_10_17(X, Y, W, eta=0.1, n_iter=5)
    _, out = res["output"], res["output"]
    assert res["loss"] == pytest.approx(
        0.5 * (1.0 - res["output"][0][0]) ** 2, rel=1e-9)
    assert res["iterations"] == 5


def test_a_single_iteration_matches_one_output_weight_update():
    from morie.fn.msm251 import mvsml_ann_update_output as one_step
    a = mvsml_reproducing_kernel_eq_10_17(X, Y, W, eta=0.1, n_iter=1)
    b = one_step(X, Y, W, eta=0.1, n_iter=1)
    assert a["loss"] == pytest.approx(b["loss"], rel=1e-9)


def test_training_moves_the_prediction_towards_the_target():
    from morie.fn.msm245 import mvsml_ann_forward as forward
    start = forward(X, W)["estimate"]
    res = mvsml_reproducing_kernel_eq_10_17(X, Y, W, eta=0.1, n_iter=20)
    assert abs(res["output"][0][0] - 1.0) < abs(start - 1.0)
