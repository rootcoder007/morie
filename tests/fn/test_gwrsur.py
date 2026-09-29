"""Tests for morie.fn.gwrsur: per-equation local WLS and the weighted residual covariance."""

import math

from morie.fn.gwrcoef import gwrcoef
from morie.fn.gwrsur import gwrsur

N = 16
P = [(float(i % 4), float(i // 4)) for i in range(N)]
X = [[(0.3 * i) % 1.7] for i in range(N)]
Y1 = [1.0 + 2.0 * X[i][0] + 0.1 * P[i][0] + 0.2 * math.sin(i) for i in range(N)]
Y2 = [0.5 - X[i][0] + 0.2 * P[i][1] + 0.3 * math.cos(2 * i) for i in range(N)]


def test_sur_equals_equationwise_gwr_and_covariance():
    r = gwrsur([Y1, Y2], X, P, 2.5, kernel="gaussian")
    b1 = gwrcoef(Y1, X, P, 2.5, kernel="gaussian")
    b2 = gwrcoef(Y2, X, P, 2.5, kernel="gaussian")
    assert max(abs(r["betas"][0][i][a] - b1[i][a]) for i in range(N) for a in range(2)) < 1e-12
    assert max(abs(r["betas"][1][i][a] - b2[i][a]) for i in range(N) for a in range(2)) < 1e-12
    e1 = [Y1[i] - b1[i][0] - b1[i][1] * X[i][0] for i in range(N)]
    e2 = [Y2[i] - b2[i][0] - b2[i][1] * X[i][0] for i in range(N)]
    i = 6
    w = [math.exp(-(math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) ** 2) / (2 * 2.5**2)) for j in range(N)]
    ref = sum(w[j] * e1[j] * e2[j] for j in range(N)) / sum(w)
    assert abs(r["local_covariance"][i][0][1] - ref) < 1e-12
