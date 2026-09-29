"""Tests for morie.fn.sptrx: the Mundlak design rebuilt by hand."""

import math

from morie.fn.spdiscrete import spatial_probit_gmm
from morie.fn.sptrx import sptrx

N, T = 6, 10
W = [[0.5 if abs(i - j) in (1, N - 1) else 0.0 for j in range(N)] for i in range(N)]
UID = list(range(N)) * T
X = [[math.sin(1.7 * k) + 0.3 * math.cos(0.4 * k)] for k in range(N * T)]
Y = [1 if X[k][0] + 0.8 * math.sin(3.3 * k + 1) + 0.3 * ((k % N) - 3) / N > 0 else 0 for k in range(N * T)]
BIG = [[W[i % N][j % N] if i // N == j // N else 0.0 for j in range(N * T)] for i in range(N * T)]


def test_design():
    xbar = [sum(X[t * N + i][0] for t in range(T)) / T for i in range(N)]
    D = [[1.0, X[k][0], xbar[k % N]] for k in range(N * T)]
    ref = spatial_probit_gmm(Y, D, BIG)
    r = sptrx(Y, X, W, UID)
    assert abs(r["rho"] - ref["rho"]) < 1e-12
    # shuffled long-format rows give the same fit
    p = [(11 * k + 5) % (N * T) for k in range(N * T)]
    tid = [k // N for k in range(N * T)]
    r2 = sptrx([Y[k] for k in p], [X[k] for k in p], W, [UID[k] for k in p], [tid[k] for k in p])
    assert abs(r2["rho"] - r["rho"]) < 1e-12
