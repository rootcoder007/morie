"""Tests for bsaclass.rangayyan_bhattacharyya (Gaussian divergence, eq. 10.117)."""

import pytest

from morie.fn.bsaclass import rangayyan_bhattacharyya


def _inv2(C):
    d = C[0][0] * C[1][1] - C[0][1] * C[1][0]
    return [[C[1][1] / d, -C[0][1] / d], [-C[1][0] / d, C[0][0] / d]]


def _mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def _tr(A):
    return A[0][0] + A[1][1]


def test_rgbhatt_basic():
    """D = 1/2 tr[(Ci - Cj)(Cj^-1 - Ci^-1)] + 1/2 tr[(Ci^-1 + Cj^-1)
    (mi - mj)(mi - mj)'], computed term by term; D_ij = D_ji."""
    m1, m2 = [0.0, 0.0], [1.0, 0.5]
    C1, C2 = [[1.0, 0.2], [0.2, 1.0]], [[2.0, 0.0], [0.0, 1.0]]
    I1, I2 = _inv2(C1), _inv2(C2)
    dC = [[C1[i][j] - C2[i][j] for j in range(2)] for i in range(2)]
    dI = [[I2[i][j] - I1[i][j] for j in range(2)] for i in range(2)]
    cov_t = 0.5 * _tr(_mm(dC, dI))
    dm = [m1[0] - m2[0], m1[1] - m2[1]]
    S = [[I1[i][j] + I2[i][j] for j in range(2)] for i in range(2)]
    mean_t = 0.5 * sum(S[i][j] * dm[i] * dm[j] for i in range(2) for j in range(2))
    r = rangayyan_bhattacharyya(m1, m2, C1, C2)
    assert r["covariance_term"] == pytest.approx(cov_t, abs=1e-14)
    assert r["mean_term"] == pytest.approx(mean_t, abs=1e-14)
    assert r["divergence"] == pytest.approx(cov_t + mean_t, abs=1e-14)
    assert rangayyan_bhattacharyya(m2, m1, C2, C1)["divergence"] == pytest.approx(r["divergence"], abs=1e-14)


def test_rgbhatt_edge():
    """Identical PDFs give zero; equal means with different covariances
    still separate, through the covariance term alone."""
    C = [[1.0, 0.3], [0.3, 2.0]]
    assert rangayyan_bhattacharyya([1.0, 2.0], [1.0, 2.0], C, C)["divergence"] == pytest.approx(0.0, abs=1e-15)
    r = rangayyan_bhattacharyya([0.0, 0.0], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [[4.0, 0.0], [0.0, 1.0]])
    # (1/2)(1 - 4)(1/4 - 1) = 1.125
    assert r["divergence"] == pytest.approx(1.125, abs=1e-15)
    assert r["mean_term"] == 0.0
