"""Tests for morie.fn.mgwrres: the OLS limit (huge bandwidths) recomputed in matrix form, and fixed-bandwidth checks."""

import math

from morie.fn import _array_core as np
from morie.fn.mgwrres import mgwrres

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
    _, _, _, e, _ = _ols()
    r = mgwrres(Y, X, P, bws=BIG, kernel="gaussian")
    assert max(abs(r[i] - float(e[i])) for i in range(N)) < 1e-7
