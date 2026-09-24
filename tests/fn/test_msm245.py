"""Verification tests for msm245.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 10, eqs. 10.1 to 10.4 p.404, the feedforward pass. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm245 import mvsml_reproducing_kernel_eq_10_4


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


def test_the_hidden_units_are_the_activated_weighted_inputs():
    # eq 10.1: V_1j = g_1(sum_i w_ji x_i)
    v, _ = _forward()
    res = mvsml_reproducing_kernel_eq_10_4(X, W)
    assert list(res["layers"][1][0]) == pytest.approx(v, rel=1e-12)
    assert v[0] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)), rel=1e-12)


def test_the_output_is_the_weighted_sum_of_the_hidden_units():
    # eq 10.3: y_l = g_3(sum_k w_lk V_2k), identity at the output
    v, out = _forward()
    res = mvsml_reproducing_kernel_eq_10_4(X, W)
    assert res["estimate"] == pytest.approx(out, rel=1e-12)
    assert res["output"][0][0] == pytest.approx(v[0] + v[1], rel=1e-12)


def test_the_first_layer_of_the_record_is_the_input_itself():
    res = mvsml_reproducing_kernel_eq_10_4(X, W)
    assert list(res["layers"][0][0]) == pytest.approx([1.0, 2.0],
                                                       rel=1e-12)


def test_zero_weights_put_every_hidden_unit_at_one_half():
    res = mvsml_reproducing_kernel_eq_10_4(X, [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0]]])
    assert list(res["layers"][1][0]) == pytest.approx([0.5, 0.5],
                                                       rel=1e-12)
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)
