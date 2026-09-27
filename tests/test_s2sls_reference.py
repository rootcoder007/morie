"""Spatial 2SLS (spatialreg::stsls)."""

import math

from morie.fn._rng import random_normal, random_uniform
from morie.fn.s2sls import spatial_two_stage_least_squares


def data():
    U = [float(u) for u in random_uniform(200, seed=51, stream=0)]
    Z = [float(z) for z in random_normal(200, seed=51, stream=1)]
    P = [[U[2 * i], U[2 * i + 1]] for i in range(40)]
    W = [[0.0] * 40 for _ in range(40)]
    for i in range(40):
        for j in sorted(range(40), key=lambda j: math.dist(P[i], P[j]))[1:5]:
            W[i][j] = 0.25
    x = [P[i][0] * 3 + Z[i] for i in range(40)]
    y = [1.0 + 2.0 * x[i] + Z[100 + i] for i in range(40)]
    Wd = [[1.0 if i != j and math.dist(a, b) <= 0.35 else 0.0 for j, b in enumerate(P)] for i, a in enumerate(P)]
    return y, [[1.0, v, v * v / 10] for v in x], W, Wd


def close(a, b, tol=1e-9):
    return all(abs(u - v) <= tol for u, v in zip(a, b))


def test_matches_stsls():
    y, X, W, Wd = data()
    r = spatial_two_stage_least_squares(y, X, W)
    # stsls(y ~ x + x2, listw = mat2listw(W, style = "W")): coef, sqrt(diag(var))
    assert close(r.value, [-0.0857735512, 1.213554624, 2.0070365699, -0.0037460142])
    assert close(r.extra["se"], [0.1182047502, 0.4844815825, 0.3165947865, 0.8815967642])
    h = spatial_two_stage_least_squares(y, X, W, robust="HC1")
    assert close(h.extra["se"], [0.1517855627, 0.5209998791, 0.2171289681, 0.6221500534])
    b = spatial_two_stage_least_squares(y, X, Wd)
    # distance-band binary weights (row sums 3 to 22): W 1 and W^2 1 enter as instruments
    assert b.extra["instruments"] == 6
    assert close(b.value, [-0.0020751248, 1.0032944399, 2.0483894929, -0.2461006652])
    assert close(b.extra["se"], [0.0066151403, 0.3763613531, 0.3221584989, 0.8497660188])
