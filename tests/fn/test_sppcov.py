"""Tests for morie.fn.sppcov: recomputed in matrix form."""

import math

from morie.fn import _array_core as np
from morie.fn.sppcov import sppcov

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


def test_pesaran_cd():
    e = [math.sin(3.1 * k) for k in range(N * T)]
    r = sppcov(e, UID, TID)
    E = [[e[t * N + i] for t in range(T)] for i in range(N)]
    D = [[v - sum(row) / T for v in row] for row in E]
    s = [math.sqrt(sum(v * v for v in row)) for row in D]
    cor = lambda i, j: sum(a * b for a, b in zip(D[i], D[j])) / (s[i] * s[j])  # noqa: E731
    cd = math.sqrt(2 * T / (N * (N - 1))) * sum(cor(i, j) for i in range(N) for j in range(i + 1, N))
    assert abs(r.statistic - cd) < 1e-12
    assert abs(r.p_value - math.erfc(abs(cd) / math.sqrt(2))) < 1e-12
    assert abs(r.extra["covariance"][0][1] - sum(E[0][t] * E[1][t] for t in range(T)) / T) < 1e-14
