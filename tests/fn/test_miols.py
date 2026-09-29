"""Tests for morie.fn.miols: recompute the exact OLS-residual moments."""

from morie.fn import _array_core as np
from morie.fn.miols import miols

N = 10
W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
X = [[1.0, float(i), (i * 7 % 5) / 2.0] for i in range(N)]
Y = [2.0, 3.5, 3.0, 5.0, 4.5, 6.0, 5.5, 8.0, 7.0, 9.5]


def test_exact_moments():
    Xa, Wa, ya = np.array(X), np.array(W), np.array(Y)
    n, k = 10, 3
    M = np.eye(n) - Xa @ np.linalg.inv(Xa.T @ Xa) @ Xa.T
    e = M @ ya
    s0 = float(Wa.sum())
    mi = n / s0 * float(e @ (Wa @ e)) / float(e @ e)
    MW = M @ Wa
    tr = float(np.trace(MW))
    E = n / s0 * tr / (n - k)
    V = (n / s0) ** 2 * (float(np.trace(MW @ M @ Wa.T)) + float(np.trace(MW @ MW)) + tr**2) / ((n - k) * (n - k + 2))
    V -= E * E
    r = miols(e.tolist(), W, X)
    assert abs(r.statistic - mi) < 1e-12
    assert abs(r.expected - E) < 1e-12
    assert abs(r.variance - V) < 1e-12


def test_intercept_default():
    e = [v - sum(Y) / N for v in Y]
    r = miols(e, W)
    assert abs(r.expected + 1.0 / (N - 1)) < 1e-12
