"""Tests for morie.fn.gwrtst: observed variances and the rank p-values."""

import math

from morie.fn.gwrcoef import gwrcoef
from morie.fn.gwrtst import gwrtst

N = 16
P = [(float(i % 4), float(i // 4)) for i in range(N)]
X = [[(0.3 * i) % 1.7] for i in range(N)]
Y = [1.0 + (2.0 + 0.3 * P[i][0]) * X[i][0] + 0.2 * math.sin(i) for i in range(N)]


def test_variance_statistic_and_pvalue():
    r = gwrtst(Y, X, P, 2.5, nsim=19, seed=4, kernel="gaussian")
    B = gwrcoef(Y, X, P, 2.5, kernel="gaussian")
    for a in range(2):
        col = [row[a] for row in B]
        m = sum(col) / N
        assert abs(r["observed_variance"][a] - sum((v - m) ** 2 for v in col) / (N - 1)) < 1e-12
        rank = 1 + sum(1 for s in r["simulated_variance"] if s[a] < r["observed_variance"][a])
        assert abs(r["p_values"][a] - (1 - rank / 20)) < 1e-15
