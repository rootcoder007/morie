"""Verification tests for msm277.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 14, eq. 14.10 p.470, the penalised sum of squares. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm277 import pensse


Y = [1.0, 2.0]
X = [[1.0, 0.0], [0.0, 1.0]]
BETA = [1.0, 1.0]
P = [[1.0, 0.0], [0.0, 1.0]]


def test_the_criterion_is_the_error_sum_plus_the_scaled_roughness():
    # eq 14.10: SSE_lambda = sum (y - mu - x' beta)^2 + lambda beta' P beta
    res = pensse(Y, X, BETA, 2.0, P, mu=0.0)
    assert list(res["fitted"]) == pytest.approx([1.0, 1.0], rel=1e-12)
    assert list(res["residuals"]) == pytest.approx([0.0, 1.0], abs=1e-12)
    assert res["sse"] == pytest.approx(1.0, rel=1e-12)
    assert res["penalty"] == pytest.approx(2.0, rel=1e-12)
    assert res["objective"] == pytest.approx(1.0 + 2.0 * 2.0, rel=1e-12)


def test_a_zero_penalty_leaves_plain_least_squares():
    res = pensse(Y, X, BETA, 0.0, P, mu=0.0)
    assert res["objective"] == pytest.approx(res["sse"], rel=1e-12)


def test_the_penalty_is_the_quadratic_form_of_the_roughness_matrix():
    res = pensse(Y, X, [2.0, 3.0], 1.0, [[2.0, 1.0], [1.0, 4.0]], mu=0.0)
    b = [2.0, 3.0]
    Pm = [[2.0, 1.0], [1.0, 4.0]]
    expected = sum(b[i] * Pm[i][j] * b[j] for i in range(2)
                   for j in range(2))
    assert res["penalty"] == pytest.approx(expected, rel=1e-12)


def test_a_larger_penalty_raises_the_criterion():
    small = pensse(Y, X, BETA, 1.0, P, mu=0.0)["objective"]
    large = pensse(Y, X, BETA, 10.0, P, mu=0.0)["objective"]
    assert large > small
