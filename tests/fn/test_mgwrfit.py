"""Tests for morie.fn.mgwrfit: the OLS limit (huge bandwidths) recomputed in matrix form, and fixed-bandwidth checks."""

import math

from morie.fn import _array_core as np
from morie.fn.mgwrfit import mgwrfit

N = 20
P = [(float(i % 5), float(i // 5)) for i in range(N)]
X = [[math.sin(i + 0.5), (0.3 * i) % 1.1 + 0.2] for i in range(N)]
Y = [1.0 + (1 + 0.2 * P[i][0]) * X[i][0] - X[i][1] + 0.1 * math.cos(3 * i) for i in range(N)]
BIG = [1e6, 1e6, 1e6]


def _ols():
    """OLS pieces: with huge bandwidths every local smoother is global, so MGWR collapses to OLS."""
    Xa = np.column_stack([np.ones(N), np.array([r[0] for r in X]), np.array([r[1] for r in X])])
    G = np.linalg.inv(Xa.T @ Xa)
    b = G @ (Xa.T @ np.array(Y))
    e = np.array(Y) - Xa @ b
    H = Xa @ G @ Xa.T
    return Xa, G, b, e, H


def test_ols_limit():
    Xa, G, b, e, H = _ols()
    r = mgwrfit(Y, X, P, bandwidths=BIG, kernel="gaussian")
    assert max(abs(r["betas"][i][k] - float(b[k])) for i in range(N) for k in range(3)) < 1e-7
    assert abs(r["trS"] - float(np.trace(H))) < 1e-6
    s2 = float(e @ e) / (N - 3)
    assert abs(r["se"][4][1] - math.sqrt(s2 * float(G[1, 1]))) < 1e-7
    assert abs(r["sigma2"] - s2) < 1e-9


def test_fixed_point():
    r = mgwrfit(Y, X, P, bandwidths=[6.0, 3.0, 8.0], kernel="gaussian")
    Xa = [[1.0] + x for x in X]
    B = r["betas"]
    for k, bw in enumerate([6.0, 3.0, 8.0]):
        part = [r["residuals"][j] + Xa[j][k] * B[j][k] for j in range(N)]
        for i in (0, 11):
            w = [math.exp(-(math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) ** 2) / (2 * bw * bw)) for j in range(N)]
            bi = sum(w[j] * Xa[j][k] * part[j] for j in range(N)) / sum(w[j] * Xa[j][k] ** 2 for j in range(N))
            assert abs(bi - B[i][k]) < 1e-7
    assert abs(sum(r["hat_diagonal"]) - r["trS"]) < 1e-12
    rss = sum(v * v for v in r["residuals"])
    ref = N * math.log(rss / N) + N * math.log(2 * math.pi) + N * (N + r["trS"]) / (N - 2 - r["trS"])
    assert abs(r["aicc"] - ref) < 1e-10
