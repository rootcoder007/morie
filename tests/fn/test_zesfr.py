"""Tests for morie.fn.zesfr: the penalised score equations vanish at the fit."""

import math

from morie.fn.zesfr import spatial_frailty

W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
REG = [i % 3 for i in range(30)]
X = [[math.sin(i)] for i in range(30)]
T = [0.5 + (i * 7 % 11) / 5 * math.exp(-0.5 * X[i][0] - 0.3 * REG[i]) for i in range(30)]
EV = [0 if i % 5 == 0 else 1 for i in range(30)]


def test_scores():
    r = spatial_frailty(T, EV, X, REG, W)
    rho, b, u = r.extra["shape"], r.extra["coefficients"][0], r.extra["frailties"]
    H = [T[i] ** rho * math.exp(X[i][0] * b + u[REG[i]]) for i in range(30)]
    assert abs(sum((EV[i] - H[i]) * X[i][0] for i in range(30))) < 1e-7
    Q = [[2 * (i == j) * (i == 1) + (i == j) * (i != 1) - W[i][j] for j in range(3)] for i in range(3)]
    for k in range(3):
        g = sum(EV[i] - H[i] for i in range(30) if REG[i] == k) - sum(Q[k][j] * u[j] for j in range(3))
        assert abs(g) < 1e-7
    ga = sum(EV[i] * (1 + rho * math.log(T[i])) - H[i] * rho * math.log(T[i]) for i in range(30))
    assert abs(ga) < 1e-7
