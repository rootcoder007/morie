"""Tests for morie.fn.tmltrn: the transported targeted estimate recomputed step by step."""

import math

from morie.fn import _array_core as np
from morie.fn.tmltrn import tmle_transportability

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


S = [1.0 if math.sin(1.9 * i) > -0.3 else 0.0 for i in range(N)]


def test_transported_ate():
    W = [[1.0] + x for x in X]
    src = [i for i in range(N) if S[i] == 1]
    tgt = [i for i in range(N) if S[i] == 0]
    pb = _logit(W, S)
    p = [_clip(_ex(w, pb), 0.025, 0.975) for w in W]
    gb = _logit([W[i] for i in src], [D[i] for i in src])
    g = [_clip(_ex(w, gb), 0.025, 0.975) for w in W]
    qb = _ols([[D[i]] + W[i] for i in src], [Y[i] for i in src])
    q1 = [sum(u * v for u, v in zip([1.0] + W[i], qb)) for i in range(N)]
    q0 = [sum(u * v for u, v in zip([0.0] + W[i], qb)) for i in range(N)]
    pt = len(tgt) / N
    odds = [(1 - v) / v for v in p]
    H = [S[i] / pt * odds[i] * (D[i] / g[i] - (1 - D[i]) / (1 - g[i])) for i in range(N)]
    qo = [q1[i] if D[i] == 1 else q0[i] for i in range(N)]
    eps = sum(H[i] * (Y[i] - qo[i]) for i in src) / sum(h * h for h in H)
    psi = sum(q1[i] + eps * odds[i] / (pt * g[i]) - q0[i] + eps * odds[i] / (pt * (1 - g[i])) for i in tgt) / len(tgt)
    r = tmle_transportability(Y, D, X, S)
    assert abs(r["estimate"] - psi) < 1e-6
