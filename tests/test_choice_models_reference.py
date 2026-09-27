"""Panel binary choice (survival::clogit, lme4::glmer) and PREFMAP bliss points."""

import math

from morie.fn._rng import random_normal, random_uniform
from morie.fn.panchc import panel_binary_choice
from morie.fn.svbliss import bliss_points


def panel():
    U = [float(v) for v in random_uniform(2000, seed=6, stream=0)]
    Z = [float(v) for v in random_normal(2000, seed=6, stream=1)]
    y, X, g = [], [], []
    for i in range(40):
        a = 0.8 * Z[i]
        for t in range(5):
            x1, x2 = Z[100 + i * 5 + t], U[500 + i * 5 + t]
            eta = -0.3 + a + 0.9 * x1 - 1.2 * x2
            y.append(1 if U[1000 + i * 5 + t] < 1 / (1 + math.exp(-eta)) else 0)
            X.append([x1, x2])
            g.append(i + 1)
    return y, X, g


def close(a, b, tol):
    return all(abs(u - v) <= tol for u, v in zip(a, b))


def test_conditional_logit_matches_clogit():
    y, X, g = panel()
    r = panel_binary_choice(y, X, g)
    # survival::clogit(y ~ x1 + x2 + strata(g), method = "exact"): coef, se, loglik
    assert close(r["coef"], [0.769142, -1.662757], 1e-6) and close(r["se"], [0.204861, 0.687196], 1e-5)
    assert abs(r["loglik"] - (-60.672473)) < 1e-5 and r["n_units"] == 35


def test_random_effects_match_glmer():
    y, X, g = panel()
    r = panel_binary_choice(y, X, g, model="re_logit")
    # lme4::glmer(y ~ x1 + x2 + (1 | g), binomial, nAGQ = 30): singular fit, sd 0
    assert close(r["coef"], [0.015379, 0.832846, -1.962072], 1e-5) and close(
        r["se"], [0.31886, 0.189852, 0.616029], 1e-5
    )
    assert r["sigma"] < 1e-3 and abs(r["loglik"] - (-107.2457)) < 1e-4
    p = panel_binary_choice(y, X, g, model="re_probit")
    assert close(p["coef"], [0.006052, 0.48501, -1.166051], 1e-5) and abs(p["loglik"] - (-107.2706)) < 1e-4


def test_random_effects_with_large_variance_match_glmer():
    U = [float(v) for v in random_uniform(2000, seed=7, stream=0)]
    Z = [float(v) for v in random_normal(2000, seed=7, stream=1)]
    y, X, g = [], [], []
    for i in range(50):
        for t in range(6):
            x1 = Z[200 + i * 6 + t]
            eta = -0.2 + 1.5 * Z[i] + 0.8 * x1
            y.append(1 if U[1000 + i * 6 + t] < 1 / (1 + math.exp(-eta)) else 0)
            X.append([x1])
            g.append(i + 1)
    r = panel_binary_choice(y, X, g, model="re_logit", n_quad=40)
    # lme4::glmer(y ~ x1 + (1 | g), binomial, nAGQ = 25)
    assert close(r["coef"], [-0.1811396, 0.8595103], 2e-5) and abs(r["sigma"] - 0.9047192) < 2e-5
    assert abs(r["loglik"] - (-183.276772)) < 1e-5


def test_bliss_points_recover_ideals():
    Z = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [0.5, 2]]
    x0 = [0.7, 0.4]
    R = [
        [5 - 2 * ((z[0] - x0[0]) ** 2 + (z[1] - x0[1]) ** 2) for z in Z],
        [1 + ((z[0] - 1) ** 2 + (z[1] - 1) ** 2) for z in Z],
    ]
    r = bliss_points(R, Z, grid=[[0.5, 0.5], [1.5, 1.5]])
    assert close(r["ideal"][0], x0, 1e-12) and abs(r["salience"][0] - 2) < 1e-12 and r["anti_ideal"] == [False, True]
    assert close(r["ideal"][1], [1.0, 1.0], 1e-12) and r["r2"] == [1.0, 1.0]
    h = r["bandwidth"]
    w = [math.exp(-0.5 * (math.dist([0.5, 0.5], z) / h) ** 2) for z in Z]
    assert abs(r["surface"][0][0] - sum(a * b for a, b in zip(w, R[0])) / sum(w)) < 1e-12
