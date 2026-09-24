"""Verification tests for km062.

Kamath, Keenan, Somers and Sorenson (2024), eq. 4.9, the merged KronA weights. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km062 import kamath_ch4_krona_tuned_weights


def test_the_merged_weights_add_the_scaled_kronecker_product():
    # Eq 4.9: W_tuned = W + s [A (x) B]
    W = [[1.0, 0.0], [0.0, 1.0]]
    A = [[1.0]]
    B = [[2.0, 0.0], [0.0, 2.0]]
    s = 0.5
    res = kamath_ch4_krona_tuned_weights(W, A, B, s)
    for i in range(2):
        for j in range(2):
            delta = s * A[0][0] * B[i][j]
            assert res["delta"][i][j] == pytest.approx(delta, rel=1e-12)
            assert res["W_tuned"][i][j] == pytest.approx(W[i][j] + delta, rel=1e-12)


def test_a_zero_scale_returns_the_original_weights():
    W = [[1.0, 2.0], [3.0, 4.0]]
    res = kamath_ch4_krona_tuned_weights(W, [[1.0]], [[9.0, 9.0], [9.0, 9.0]], 0.0)
    for i in range(2):
        for j in range(2):
            assert res["W_tuned"][i][j] == pytest.approx(W[i][j], rel=1e-12)


def test_the_merged_shape_matches_the_base_weights():
    res = kamath_ch4_krona_tuned_weights([[1.0, 0.0], [0.0, 1.0]], [[1.0]], [[1.0, 0.0], [0.0, 1.0]], 1.0)
    assert tuple(res["shape"]) == (2, 2)
