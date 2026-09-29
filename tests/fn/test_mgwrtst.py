"""Tests for morie.fn.mgwrtst: observed variances and rank p-values recomputed."""

import math

from morie.fn.mgwrcof import mgwrcof
from morie.fn.mgwrtst import mgwrtst

N = 20
P = [(float(i % 5), float(i // 5)) for i in range(N)]
X = [[math.sin(i + 0.5), (0.3 * i) % 1.1 + 0.2] for i in range(N)]
Y = [1.0 + (1 + 0.2 * P[i][0]) * X[i][0] - X[i][1] + 0.1 * math.cos(3 * i) for i in range(N)]


def test_statistic_and_pvalues():
    b = [6.0, 3.0, 8.0]
    r = mgwrtst(Y, X, P, nsim=9, bandwidths=b, seed=2, kernel="gaussian", threshold=1e-12)
    B = mgwrcof(Y, X, P, bws=b, kernel="gaussian")
    for k in range(3):
        col = [row[k] for row in B]
        m = sum(col) / N
        assert abs(r["observed_variance"][k] - sum((v - m) ** 2 for v in col) / (N - 1)) < 1e-9
        rank = 1 + sum(1 for s in r["simulated_variance"] if s[k] < r["observed_variance"][k])
        assert abs(r["p_values"][k] - (1 - rank / 10)) < 1e-15
