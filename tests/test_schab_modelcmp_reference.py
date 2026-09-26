"""Nested spatial covariance models (eqs 6.57-6.60) against nlme::gls with corExp."""

import functools
import math

from morie.fn.spcmp import spatial_covariance_comparison

N = 40
CO = [(10 * ((i * 0.618034) % 1), 10 * ((i * 0.414214 + 0.3) % 1)) for i in range(N)]
W = [0.5 + 0.1 * x for x, _ in CO]
Z = [
    1
    + 0.4 * wi
    + math.sin(0.9 * x)
    + math.cos(0.7 * y)
    + 0.5 * math.sin(0.6 * x + 1.1 * y)
    + 0.8 * (((97 * i) % 13) - 6) / 6
    for i, ((x, y), wi) in enumerate(zip(CO, W))
]
X = [[1.0, wi] for wi in W]
# -2 logLik of gls(z ~ w, correlation = corExp(form = ~ x + y, nugget = TRUE / FALSE)) and gls(z ~ w)
NLME = {
    "ml": (102.72806151208869, 102.72806149996285, 119.18462546682153),
    "reml": (100.36362758924545, 100.36362757931411, 120.09536318618083),
}


@functools.cache
def fit(method):
    return spatial_covariance_comparison(CO, Z, X, "exponential", method)


def test_fits_match_nlme_and_aic_follows_6_59_6_60():
    for method, (full, nonug, ind) in NLME.items():
        r = fit(method)
        f = r["fits"]
        assert abs(f["independent"]["neg2loglik"] - ind) < 1e-10
        assert abs(f["no_nugget"]["neg2loglik"] - nonug) < 1e-8
        assert abs(f["nugget"]["neg2loglik"] - full) < 1e-7
        assert abs(r["lrt_spatial"] - (ind - nonug)) < 1e-7
        extra = 2 if method == "ml" else 0
        assert f["independent"]["aic"] == f["independent"]["neg2loglik"] + 2 * (1 + extra)
        assert f["no_nugget"]["aic"] == f["no_nugget"]["neg2loglik"] + 2 * (2 + extra)


def test_nugget_on_the_boundary_halves_the_p_value():
    r = fit("reml")
    assert r["lrt_nugget"] < 1e-6
    assert abs(r["p_nugget"] - 0.5 * r["p_nugget_naive"]) < 1e-15
    assert abs(r["p_spatial"] - math.erfc(math.sqrt(r["lrt_spatial"] / 2))) < 1e-15
