"""Tests for morie.fn.tmlmda: the targeted estimate recomputed step by step."""

import math

from morie.fn import _array_core as np
from morie.fn.tmlmda import tmle_missing_data

N = 40
X = [[math.sin(0.7 * i), math.cos(1.3 * i)] for i in range(N)]
D = [1.0 if math.sin(2.1 * i + X[i][0]) > 0 else 0.0 for i in range(N)]
Y = [1.0 + 2.0 * D[i] + X[i][0] - 0.5 * X[i][1] + 0.3 * math.sin(5 * i) for i in range(N)]


def _logit(rows, y):
    """Logistic regression by Newton-Raphson to convergence."""
    A = np.array(rows)
    b = np.zeros(len(rows[0]))
    for _ in range(60):
        p = [1 / (1 + math.exp(-float(v))) for v in (A @ b).tolist()]
        W = np.diag(np.array([q * (1 - q) for q in p]))
        b = b + np.linalg.inv(A.T @ W @ A) @ (A.T @ (np.array(y) - np.array(p)))
    return b.tolist()


def _ols(rows, y):
    A = np.array(rows)
    return (np.linalg.inv(A.T @ A) @ (A.T @ np.array(y))).tolist()


def _ex(r, b):
    return 1 / (1 + math.exp(-sum(u * v for u, v in zip(r, b))))


def _clip(v, lo, hi):
    return min(max(v, lo), hi)


MISS = [1.0 if math.cos(3.7 * i) > 0.6 else 0.0 for i in range(N)]


def test_targeted_ate():
    W = [[1.0] + x for x in X]
    dl = [1 - m for m in MISS]
    gb = _logit(W, D)
    g = [_clip(_ex(w, gb), 0.025, 0.975) for w in W]
    pb = _logit([[D[i]] + W[i] for i in range(N)], dl)

    def pi(i, a):
        return _clip(_ex([a] + W[i], pb), 0.025, 1.0)

    obs = [i for i in range(N) if dl[i] == 1]
    qb = _ols([[D[i]] + W[i] for i in obs], [Y[i] for i in obs])

    def q(i, a):
        return sum(u * v for u, v in zip([a] + W[i], qb))

    H = [dl[i] / pi(i, D[i]) * (D[i] / g[i] - (1 - D[i]) / (1 - g[i])) for i in range(N)]
    eps = sum(H[i] * (Y[i] - q(i, D[i])) for i in obs) / sum(h * h for h in H)
    psi = sum(q(i, 1) + eps / (g[i] * pi(i, 1)) - q(i, 0) + eps / ((1 - g[i]) * pi(i, 0)) for i in range(N)) / N
    r = tmle_missing_data(Y, D, X, MISS)
    assert abs(r["eps"] - eps) < 1e-6
    assert abs(r["estimate"] - psi) < 1e-6
