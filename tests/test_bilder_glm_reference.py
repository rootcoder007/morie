"""Bilder & Loughin eqs 2.12-2.13, 2.18, 2.23, 3.9, 6.2, Secs 4.1 and 5.2.1 against R and first principles."""

import math

from morie.fn._rrng_core import qchisq, qnorm
from morie.fn.glmprofci import _irls, glmprofci
from morie.fn.glmstdres import glmstdres
from morie.fn.invpredci import invpredci
from morie.fn.orint import orint
from morie.fn.poisexactci import poisexactci
from morie.fn.rogangladen import rogangladen

DIST = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
SUCC = [19, 18, 17, 15, 12, 10, 8, 5, 3, 1]
CNT = [2, 3, 6, 7, 8, 9, 10, 12, 15, 20, 22]


def test_profile_interval_solves_the_lr_equation():
    X = [[1.0, float(d)] for d in DIST]
    r = glmprofci(SUCC, X, 1, trials=[20] * 10)
    _, _, ll, _ = _irls(SUCC, X, "binomial", [20.0] * 10, [0.0] * 10)
    for b in r["ci"]:
        _, _, llb, _ = _irls(SUCC, [[1.0]] * 10, "binomial", [20.0] * 10, [d * b for d in DIST])
        assert abs(2 * (ll - llb) - qchisq(0.95, 1)) < 1e-8  # (2.13) holds at both limits
    assert abs(r["estimate"] + 0.116088969225002) < 1e-11  # glm
    assert (
        abs(r["ci"][0] + 0.149231637058244) < 1e-9 and abs(r["ci"][1] + 0.0869796115891111) < 1e-9
    )  # R uniroot on glm offset
    p = glmprofci(CNT, [[1, x, x * x / 10] for x in range(11)], 2, family="poisson")
    assert abs(p["ci"][0] + 0.299712466652543) < 1e-9 and abs(p["ci"][1] - 0.169959611101528) < 1e-9


def test_standardized_residuals_match_rstandard():
    r = glmstdres(SUCC, [[1, d] for d in DIST], trials=[20] * 10)
    assert [round(v, 7) for v in r["standardized"][:4]] == [0.0789306, -0.1405998, 0.0438949, -0.0671914]
    assert abs(sum(r["hat"]) - 2) < 1e-10  # trace of the hat matrix is p
    p = glmstdres(CNT, [[1, x, x * x / 10] for x in range(11)], family="poisson")
    assert abs(p["standardized"][0] + 0.643523614025) < 1e-9 and abs(p["standardized"][3] - 0.50333409692) < 1e-9
    e = (CNT[0] - p["fitted"][0]) / math.sqrt(p["fitted"][0])
    assert abs(p["pearson"][0] - e) < 1e-15 and abs(p["standardized"][0] - e / math.sqrt(1 - p["hat"][0])) < 1e-15


def test_odds_ratio_inverse_prediction_poisson_and_misclassification():
    r = orint(0.5, 0.1, 2.0, 0.01, 0.001, -0.002, c=2)
    v = 0.01 + 4 * 0.001 - 0.008
    z = qnorm(0.975)
    assert (
        abs(r["odds_ratio"] - math.exp(1.4)) < 1e-12 and abs(r["ci"][0] - math.exp(1.4 - 2 * z * math.sqrt(v))) < 1e-12
    )
    q = invpredci(-4.0, 2.0, 0.25, 0.04, -0.09)
    for x in q["ci"]:  # the limits satisfy (2.23) with equality
        assert abs(abs(-4 + 2 * x) / math.sqrt(0.25 + x * x * 0.04 - 0.18 * x) - z) < 1e-10
    assert q["ci"][0] < 2.0 < q["ci"][1]
    lo, hi = poisexactci(7, 3)["ci"]
    assert abs(lo - 0.938121017173289) < 1e-12 and abs(hi - 4.807558453900793) < 1e-12  # poisson.test(7, 3)
    assert poisexactci(0, 4)["ci"] == (0.0, qchisq(0.975, 2) / 8)
    g = rogangladen(30, 200, 0.9, 0.95)
    lik = lambda p: 30 * math.log(0.9 * p + 0.05 * (1 - p)) + 170 * math.log(1 - 0.9 * p - 0.05 * (1 - p))  # noqa: E731
    assert lik(g["estimate"]) >= max(lik(g["estimate"] + d) for d in (-1e-4, 1e-4))  # maximises (6.2)
    assert rogangladen(2, 200, 0.9, 0.95)["estimate"] == 0.0  # apparent rate below 1 - Sp truncates at 0
