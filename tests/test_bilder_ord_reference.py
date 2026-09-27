"""Proportional odds ML (vs MASS::polr), Rao-Scott corrections (6.12), GLM deviance and Pearson X^2."""

import math

from morie.fn.glmstdres import glmstdres
from morie.fn.polrfit import polrfit
from morie.fn.raoscott import raoscott


def lcg(seed):
    xs = seed
    while True:
        xs = xs * 16807 % 2147483647
        yield xs / 2147483647


def test_proportional_odds_matches_polr():
    u = lcg(4242)
    x, y = [], []
    for _ in range(200):
        a = 4 * next(u) - 2
        b = next(u)
        e = math.log(next(u) / (1 - next(u)))
        v = 0.8 * a - 1.2 * b + e
        x.append([a, b])
        y.append(0 if v < -1 else 1 if v < 0.5 else 2 if v < 1.5 else 3)
    r = polrfit(y, x)
    for got, ref in zip(
        r["intercepts"] + r["beta"], (-1.786386446619, 0.717897594557, 1.535142493703, -1.46834013453, 2.33457023102)
    ):
        assert abs(got - ref) < 1e-9  # MASS::polr zeta and -coef
    assert abs(r["loglik"] + 187.819396146) < 1e-8
    assert abs(r["se_beta"][0] - 0.1706043721) < 1e-6  # polr's numerical Hessian


def test_rao_scott_forms():
    d = [0.8, 1.4, 2.1, 0.5]
    r = raoscott(9.3, d, kappa=30)
    db = sum(d) / 4
    c2 = sum((v - db) ** 2 for v in d) / (4 * db * db)
    assert abs(r["rs2"] - 9.3 / (db * (1 + c2))) < 1e-14 and abs(r["rs2_df"] - 4 / (1 + c2)) < 1e-14
    s2 = sum(v * v for v in d)
    assert abs(r["rs2"] - 4 * db * 9.3 / s2) < 1e-12 and abs(r["rs2_df"] - sum(d) ** 2 / s2) < 1e-12  # Satterthwaite
    assert abs(r["f_tr"] - r["rs1"] / 4) < 1e-15 and abs(r["f_df"][1] - 30 * r["rs2_df"]) < 1e-12


def test_deviance_and_pearson_statistic():
    dist = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    succ = [19, 18, 17, 15, 12, 10, 8, 5, 3, 1]
    r = glmstdres(succ, [[1, d] for d in dist], trials=[20] * 10)
    mu = r["fitted"]
    dev = 2 * sum(
        (w * math.log(w / m) if w else 0) + ((20 - w) * math.log((20 - w) / (20 - m)) if w < 20 else 0)
        for w, m in zip(succ, mu)
    )
    assert abs(r["deviance"] - dev) < 1e-12 and abs(r["pearson_chisq"] - sum(e * e for e in r["pearson"])) < 1e-12
    assert r["df"] == 8
    assert (
        abs(r["deviance"] - 0.76421933017812904) < 1e-9 and abs(r["pearson_chisq"] - 0.71358727577615844) < 1e-9
    )  # glm deviance, sum of squared Pearson residuals
