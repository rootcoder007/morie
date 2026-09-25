"""Tests for ssmkrn.s4_ssm_kernel (K_l = C A^l B, Gu, Goel & Re 2022)."""

import pytest

from morie.fn._unclrcore import ssmconv
from morie.fn.ssmkrn import s4_ssm_kernel, ssmk


def test_ssmkrn_basic():
    """K_l = C A^l B term by term, and convolving with K equals running
    the state recurrence h_t = A h_{t-1} + B x_t, y_t = C h_t."""
    A = [[0.5, 0.1], [-0.2, 0.3]]
    B = [1.0, -0.5]
    C = [0.3, 2.0]
    L = 6
    r = s4_ssm_kernel(A, B, C, L)
    v, K = B[:], []
    for _ in range(L):
        K.append(C[0] * v[0] + C[1] * v[1])
        v = [A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1]]
    assert r["K"] == pytest.approx(K, abs=1e-15)
    assert r["L"] == 6 and r["state_dim"] == 2
    x = [1.0, 0.0, -2.0, 0.5, 3.0, 1.0]
    h, y = [0.0, 0.0], []
    for xt in x:
        h = [A[0][0] * h[0] + A[0][1] * h[1] + B[0] * xt, A[1][0] * h[0] + A[1][1] * h[1] + B[1] * xt]
        y.append(C[0] * h[0] + C[1] * h[1])
    assert ssmconv(r["K"], x) == pytest.approx(y, abs=1e-13)


def test_ssmkrn_edge():
    """A zero-length kernel is empty; mismatched B or C raise."""
    assert ssmk([[1.0]], [1.0], [1.0], 0)["K"] == []
    with pytest.raises(ValueError):
        ssmk([[1.0, 0.0], [0.0, 1.0]], [1.0], [1.0, 1.0], 3)
