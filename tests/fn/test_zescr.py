"""Tests for morie.fn.zescr: score equations and the cure fractions."""

import math

from morie.fn.zescr import spatial_cure_rate

W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
REG = [i % 3 for i in range(36)]
X = [[math.sin(i)] for i in range(36)]
T = [5.0 if i % 4 == 0 else 0.2 + (i * 7 % 11) / 8 for i in range(36)]
EV = [0 if (i % 4 == 0 or i % 9 == 1) else 1 for i in range(36)]


def test_scores_and_cure():
    r = spatial_cure_rate(T, EV, X, REG, W)
    rho, c, b, u = r.extra["shape"], r.extra["log_scale"], r.extra["coefficients"][0], r.extra["regional_effects"]
    th = [math.exp(X[i][0] * b + u[REG[i]]) for i in range(36)]
    G = [math.exp(c + rho * math.log(T[i])) for i in range(36)]
    e = [EV[i] - th[i] * (1 - math.exp(-G[i])) for i in range(36)]
    assert abs(sum(e[i] * X[i][0] for i in range(36))) < 1e-7
    gc = sum(EV[i] * (1 - G[i]) - th[i] * math.exp(-G[i]) * G[i] for i in range(36))
    assert abs(gc) < 1e-7
    assert max(abs(r.extra["cure_fraction"][i] - math.exp(-th[i])) for i in range(36)) < 1e-12
