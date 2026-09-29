"""Tests for morie.fn.cargmm: the moment estimator e'We / e'W^2 e."""

from morie.fn.cargmm import cargmm

N = 8
W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
Y = [1.0, 2.0, 2.5, 4.5, 4.0, 3.2, 2.1, 2.9]


def test_moment_estimator():
    m = sum(Y) / N
    e = [v - m for v in Y]
    we = [sum(W[i][j] * e[j] for j in range(N)) for i in range(N)]
    wwe = [sum(W[i][j] * we[j] for j in range(N)) for i in range(N)]
    ref = sum(a * b for a, b in zip(e, we)) / sum(a * b for a, b in zip(e, wwe))
    assert abs(cargmm(Y, W).statistic - ref) < 1e-12
    # the estimating equation sum (e - rho We) We = 0 holds
    rho = cargmm(Y, W).statistic
    assert abs(sum((e[i] - rho * we[i]) * we[i] for i in range(N))) < 1e-12
