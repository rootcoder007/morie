"""Verification tests for km061.

Kamath, Keenan, Somers and Sorenson (2024), eq. 4.8, the KronA layer output. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km061 import kamath_ch4_krona_output


def test_the_krona_output_is_the_base_plus_the_scaled_kronecker_term():
    # Eq 4.8: Y = X W + s X [A (x) B] = X (W + s [A (x) B])
    X = [[1.0, 0.0]]
    W = [[1.0, 0.0], [0.0, 1.0]]
    A = [[1.0]]
    B = [[2.0, 0.0], [0.0, 2.0]]
    s = 0.5
    res = kamath_ch4_krona_output(X, W, A, B, s)
    # A (x) B with A a one-by-one is just A[0][0] * B
    kron = [[A[0][0] * B[i][j] for j in range(2)] for i in range(2)]
    base = [[sum(X[0][k] * W[k][j] for k in range(2)) for j in range(2)]]
    adapter = [[s * sum(X[0][k] * kron[k][j] for k in range(2)) for j in range(2)]]
    for j in range(2):
        assert res["base"][0][j] == pytest.approx(base[0][j], rel=1e-12)
        assert res["adapter_term"][0][j] == pytest.approx(adapter[0][j], rel=1e-12)
        assert res["Y"][0][j] == pytest.approx(base[0][j] + adapter[0][j], rel=1e-12)


def test_a_zero_scale_leaves_the_base_output_untouched():
    X = [[1.0, 2.0]]
    W = [[1.0, 0.0], [0.0, 1.0]]
    res = kamath_ch4_krona_output(X, W, [[1.0]], [[3.0, 0.0], [0.0, 3.0]], 0.0)
    assert res["Y"][0][0] == pytest.approx(1.0, rel=1e-12)
    assert res["Y"][0][1] == pytest.approx(2.0, rel=1e-12)


def test_distributivity_holds_so_the_adapter_can_be_merged():
    # Eq 4.8 equals Eq 4.9 applied to X, which is why KronA is free at
    # inference: the two forms must agree
    X = [[1.5, -0.5]]
    W = [[1.0, 0.25], [0.5, 1.0]]
    A = [[1.0]]
    B = [[2.0, 0.0], [0.0, 2.0]]
    s = 0.25
    res = kamath_ch4_krona_output(X, W, A, B, s)
    merged = [[W[i][j] + s * A[0][0] * B[i][j] for j in range(2)] for i in range(2)]
    direct = [sum(X[0][k] * merged[k][j] for k in range(2)) for j in range(2)]
    for j in range(2):
        assert res["Y"][0][j] == pytest.approx(direct[j], rel=1e-12)
