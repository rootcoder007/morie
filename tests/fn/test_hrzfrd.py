"""Tests for hrzfrd.horowitz_fredholm_eq: the Tikhonov normal equations."""

from morie.fn import _array_core as np
from morie.fn.hrzfrd import horowitz_fredholm_eq

K = [[1.0, 0.5, 0.2], [0.5, 1.0, 0.5], [0.2, 0.5, 1.0], [0.1, 0.3, 0.8]]
G = [0.5, -1.0, 2.0]


def test_recovery_and_formula():
    m = [sum(K[i][j] * G[j] for j in range(3)) for i in range(4)]
    r = horowitz_fredholm_eq(m, K, alpha=1e-12)
    assert max(abs(a - b) for a, b in zip(r["g_hat"], G)) < 1e-8
    w = [0.5, 1.0, 0.25]
    r = horowitz_fredholm_eq(m, K, alpha=0.3, weights=w)
    A = np.array([[K[i][j] * w[j] for j in range(3)] for i in range(4)])
    ref = np.linalg.inv(A.T @ A + 0.3 * np.eye(3)) @ (A.T @ np.array(m))
    assert max(abs(float(ref[j]) - r["g_hat"][j]) for j in range(3)) < 1e-12
