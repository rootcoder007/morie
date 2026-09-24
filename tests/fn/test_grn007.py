"""Verification tests for grn007.

Geron (2023), *Hands-On Machine Learning with Scikit-Learn, Keras and
TensorFlow*, 3rd edition, ch 4, the mean squared error gradient vector. Expected values are recomputed in
the test body and the docstring's own worked value is asserted too.
"""

import math

import pytest

from morie.fn.grn007 import geron_ch4_mse_gradient_vector


X = [[1.0, 1.0], [1.0, 2.0]]
Y = [1.0, 2.0]


def test_the_gradient_is_the_design_transpose_times_the_residual():
    # (2/m) X' (X theta - y), here with theta at the origin
    res = geron_ch4_mse_gradient_vector(X, Y, [0.0, 0.0])
    expected = [(2.0 / 2.0) * sum(X[i][j] * (0.0 - Y[i])
                                  for i in range(2)) for j in range(2)]
    assert list(res["gradient"]) == pytest.approx(expected, rel=1e-12)
    assert list(res["gradient"]) == pytest.approx([-3.0, -5.0],
                                                   rel=1e-12)


def test_the_gradient_vanishes_at_the_least_squares_optimum():
    # theta = [0, 1] fits both rows exactly
    res = geron_ch4_mse_gradient_vector(X, Y, [0.0, 1.0])
    assert list(res["gradient"]) == pytest.approx([0.0, 0.0], abs=1e-12)


def test_the_gradient_is_an_average_so_duplicating_the_data_keeps_it():
    small = geron_ch4_mse_gradient_vector(X, Y, [0.0, 0.0])["gradient"]
    big = geron_ch4_mse_gradient_vector(X + X, Y + Y, [0.0, 0.0])["gradient"]
    assert list(big) == pytest.approx(list(small), rel=1e-12)


def test_overshooting_flips_the_sign_of_the_gradient():
    under = geron_ch4_mse_gradient_vector(X, Y, [0.0, 0.0])["gradient"]
    over = geron_ch4_mse_gradient_vector(X, Y, [0.0, 2.0])["gradient"]
    assert all(a < 0 for a in under)
    assert all(b > 0 for b in over)
