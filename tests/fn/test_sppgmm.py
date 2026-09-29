"""Tests for morie.fn.sppgmm: recomputed in matrix form."""

import math

from morie.fn import _array_core as np
from morie.fn.sppgmm import sppgmm

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


def test_moment_fit_and_fgls():
    r = sppgmm(Y, X, W, TID, UID)
    lam = r["lambda"]
    Xa = np.column_stack([np.ones(N * T)] + [np.array([x[c] for x in X]) for c in range(2)])
    u = np.array(Y) - Xa @ (np.linalg.inv(Xa.T @ Xa) @ (Xa.T @ np.array(Y)))
    ub = np.array(_lag(u.tolist()))
    e = (u - lam * ub).tolist()
    eb = (ub - lam * np.array(_lag(ub.tolist()))).tolist()
    d = N * (T - 1)
    ew, ebw = _within(e), _within(eb)
    s2 = r["sigma2_nu"]
    m1 = sum(a * b for a, b in zip(e, ew)) / d - s2
    m2 = sum(a * b for a, b in zip(eb, ebw)) / d - s2 * sum(v * v for row in W for v in row) / N
    m3 = sum(a * b for a, b in zip(eb, ew)) / d
    assert abs(m1 * m1 + m2 * m2 + m3 * m3 - r["moment_objective"]) < 1e-12
    eu = [sum(e[t * N + i] for t in range(T)) / T for i in range(N)]
    assert abs(r["sigma2_1"] - T * sum(v * v for v in eu) / N) < 1e-12

    def om(v):
        w = _within(v)
        return [a / math.sqrt(s2) + (b - a) / math.sqrt(r["sigma2_1"]) for a, b in zip(w, v)]

    ys = om([a - lam * b for a, b in zip(Y, _lag(Y))])
    cols = [[1.0] * (N * T)] + [[x[c] for x in X] for c in range(2)]
    Xs = np.column_stack([np.array(om([a - lam * b for a, b in zip(c, _lag(c))])) for c in cols])
    b = np.linalg.inv(Xs.T @ Xs) @ (Xs.T @ np.array(ys))
    assert max(abs(float(b[k]) - r["coefficients"][k]) for k in range(3)) < 1e-9
