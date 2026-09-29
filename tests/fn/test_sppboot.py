"""Tests for morie.fn.sppboot: recomputed in matrix form."""

import math

from morie.fn import _array_core as np
from morie.fn.sppboot import sppboot

N, T = 5, 4
W = [[0.5 if abs(i - j) in (1, N - 1) else 0.0 for j in range(N)] for i in range(N)]
TID = [t for t in range(T) for _ in range(N)]
UID = [u for _ in range(T) for u in range(N)]
X = [[math.sin(1.3 * k) + 0.1 * k, math.cos(0.7 * k)] for k in range(N * T)]
Y = [1.0 + 0.8 * X[k][0] - 0.5 * X[k][1] + 0.3 * math.cos(2.1 * k) + 0.2 * (k % N) for k in range(N * T)]
PERM = [(7 * k + 3) % (N * T) for k in range(N * T)]


def _shuffled():
    """The same panel with its long-format rows permuted."""
    return [Y[k] for k in PERM], [X[k] for k in PERM], [TID[k] for k in PERM], [UID[k] for k in PERM]


def _within(v):
    m = [sum(v[t * N + i] for t in range(T)) / T for i in range(N)]
    return [v[t * N + i] - m[i] for t in range(T) for i in range(N)]


def _lag(v):
    return [sum(W[i][j] * v[t * N + j] for j in range(N)) for t in range(T) for i in range(N)]


def _cll(rho, yt, xt):
    """Concentrated FE lag log-likelihood -NT/2 log SSE + T log|I - rho W|."""
    Xa = np.column_stack([np.array(c) for c in xt])
    z = np.array([a - rho * b for a, b in zip(yt, _lag(yt))])
    e = z - Xa @ (np.linalg.inv(Xa.T @ Xa) @ (Xa.T @ z))
    A = np.eye(N) - rho * np.array(W)
    return -N * T / 2 * math.log(float(e @ e)) + T * math.log(abs(float(np.linalg.det(A))))


def test_first_replicate():
    from morie.fn._rng import random_uniform
    from morie.fn.sppfe import sppfe

    r = sppboot(Y, X, W, TID, UID, B=3, seed=5)
    f = sppfe(Y, X, W, TID, UID)
    assert abs(r["rho"] - f["rho"]) < 1e-15
    res = f["residuals"]
    m = sum(res) / (N * T)
    u = random_uniform(N * T, seed=5, stream=0)
    es = [res[min(N * T - 1, int(math.floor(float(v) * N * T)))] - m for v in u]
    xt = [_within([x[c] for x in X]) for c in range(2)]
    mu = [sum(xt[c][k] * f["coefficients"][c] for c in range(2)) for k in range(N * T)]
    Ai = np.linalg.inv(np.eye(N) - f["rho"] * np.array(W))
    ys = []
    for t in range(T):
        ys += (Ai @ np.array([mu[t * N + i] + es[t * N + i] for i in range(N)])).tolist()
    g = sppfe(ys, [[xt[0][k], xt[1][k]] for k in range(N * T)], W, TID, UID)
    assert abs(r["rho_draws"][0] - g["rho"]) < 1e-12
    rd = r["rho_draws"]
    mm = sum(rd) / 3
    assert abs(r["rho_se"] - math.sqrt(sum((v - mm) ** 2 for v in rd) / 2)) < 1e-15
