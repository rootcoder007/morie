"""Spatial lag / error / SAC ML (spatialreg lagsarlm, errorsarlm, sacsarlm, method = "LU")."""

import math

from morie.fn._rng import random_normal, random_uniform
from morie.fn.sarreg import spatial_regression_ml


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
    return y, [[1.0, v] for v in x], W


def close(a, b, tol):
    return all(abs(u - v) <= tol for u, v in zip(a, b))


def test_lag_and_error_match_spatialreg():
    y, X, W = data()
    r = spatial_regression_ml(y, X, W, model="lag")
    assert abs(r.extra["rho"] - (-0.1184623507166)) < 1e-7 and close(r.value, [1.3121577442088, 2.0232715384989], 1e-7)
    assert abs(r.extra["loglik"] - (-52.1979401739328)) < 1e-9 and abs(r.extra["aic"] - 112.3958803478657) < 1e-8
    assert close(r.extra["se"][:3], [0.3772750619259, 0.1327597567211, 0.0991681511581], 1e-8)
    e = spatial_regression_ml(y, X, W, model="error")
    assert abs(e.extra["lambda"] - (-0.300725704492)) < 1e-7 and abs(e.extra["loglik"] - (-52.299911639139)) < 1e-9
    assert close(e.extra["se"][:3], [0.193590739692, 0.110512114329, 0.26638262466], 1e-8)


def test_sac_matches_sacsarlm():
    y, X, W = data()
    s = spatial_regression_ml(y, X, W, model="sac")
    assert abs(s.extra["rho"] - (-0.0860235065878)) < 1e-6 and abs(s.extra["lambda"] - (-0.2015844059734)) < 1e-6
    assert abs(s.extra["loglik"] - (-51.9593749974537)) < 1e-9 and abs(s.extra["aic"] - 113.9187499949073) < 1e-8
    # numerical Hessians on both sides: agreement to about 1e-3 relative
    assert close(s.extra["se"][:4], [0.3444010078623, 0.1397112086575, 0.105073190512, 0.2933415676024], 5e-4)
