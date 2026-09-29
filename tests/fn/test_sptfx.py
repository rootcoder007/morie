"""Tests for morie.fn.sptfx: the dummy-variable design rebuilt by hand."""

import math

from morie.fn.spdiscrete import spatial_probit_gmm
from morie.fn.sptfx import sptfx

N, T = 6, 10
W = [[0.5 if abs(i - j) in (1, N - 1) else 0.0 for j in range(N)] for i in range(N)]
UID = list(range(N)) * T
X = [[math.sin(1.7 * k) + 0.3 * math.cos(0.4 * k)] for k in range(N * T)]
Y = [1 if X[k][0] + 0.8 * math.sin(3.3 * k + 1) + 0.3 * ((k % N) - 3) / N > 0 else 0 for k in range(N * T)]
BIG = [[W[i % N][j % N] if i // N == j // N else 0.0 for j in range(N * T)] for i in range(N * T)]


def test_design():
    D = [[1.0, X[k][0]] + [1.0 if k % N == g else 0.0 for g in range(1, N)] for k in range(N * T)]
    lx = [sum(BIG[i][j] * X[j][0] for j in range(N * T)) for i in range(N * T)]
    ref = spatial_probit_gmm(Y, D, BIG, Z=[d + [v] for d, v in zip(D, lx)])
    r = sptfx(Y, X, W, UID)
    assert abs(r["rho"] - ref["rho"]) < 1e-12
    assert len(r["coefficients"]) == 2 + N - 1
