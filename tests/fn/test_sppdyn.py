"""Tests for morie.fn.sppdyn: row matching and the estimating equations recomputed."""

import math

from morie.fn import _array_core as np
from morie.fn.sppdyn import sppdyn

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


def test_dynamic_design():
    r = sppdyn(Y, X, W, TID, UID)
    # the model is the FE lag model on periods 2..T with y_{t-1}, W y_{t-1} and X_t as regressors
    yy = Y[N:]
    lagy = Y[: N * (T - 1)]
    wl = _lag(Y)[: N * (T - 1)]
    rows = [[lagy[k], wl[k]] + X[N + k] for k in range(N * (T - 1))]
    Tn = T - 1

    def within(v):
        m = [sum(v[t * N + i] for t in range(Tn)) / Tn for i in range(N)]
        return [v[t * N + i] - m[i] for t in range(Tn) for i in range(N)]

    yt = within(yy)
    Xa = np.column_stack([np.array(within([r_[c] for r_ in rows])) for c in range(4)])
    wy = [sum(W[i][j] * yt[t * N + j] for j in range(N)) for t in range(Tn) for i in range(N)]
    z = np.array([a - r["rho"] * b for a, b in zip(yt, wy)])
    b = np.linalg.inv(Xa.T @ Xa) @ (Xa.T @ z)
    assert max(abs(float(b[k]) - r["beta"][k]) for k in range(4)) < 1e-9
