"""Gibbs MPLE (spatstat ppm on the same quadrature) and sequential inhibition."""

import math

from morie.fn._rng import random_uniform
from morie.fn.gibbsp import gibbs_pseudolikelihood
from morie.fn.seqinh import sequential_inhibition


def pattern():
    U = [float(u) for u in random_uniform(120, seed=21, stream=0)]
    return [[2 * U[2 * i], U[2 * i + 1]] for i in range(60)]


def test_gibbs_mple_matches_ppm():
    P = pattern()
    # ppm(quadscheme(X, D, method = "grid", ntile = c(12, 12)) ~ 1, I, correction = "none"):
    # coef and sqrt(diag(vcov(fit, hessian = TRUE)))
    cases = [
        (dict(interaction="strauss", r=0.1), [3.6694539943, -0.441687331], [0.16172812, 0.19482627]),
        (dict(interaction="geyer", r=0.1, sat=2), [3.658567001, -0.2152837095], [0.16317816, 0.09980203]),
        (dict(interaction="softcore", kappa=0.5), [3.3879516916, -0.6309547953], [0.13038334, 0.54554565]),
        (
            dict(interaction="diggle_gratton", delta=0.005, rho=0.1),
            [3.4691805813, 0.2088573719],
            [0.14921173, 0.25361331],
        ),
    ]
    for kw, coef, se in cases:
        r = gibbs_pseudolikelihood(P, (0, 2, 0, 1), **kw)
        assert all(abs(a - b) < 1e-8 for a, b in zip(r.value, coef))
        assert all(abs(a - b) < 1e-8 for a, b in zip(r.extra["se"], se))


def test_sequential_inhibition():
    r = sequential_inhibition(0.2, (0, 1, 0, 1), n=5, seed=1)
    assert r.extra["proposals"] == 8 and len(r.value) == 5
    # SequentialInhibition(0.2, c(0, 1, 0, 1), n = 5, seed = 1)$points[1, ]
    assert abs(r.value[0][0] - 0.8902591728838161) < 1e-15 and abs(r.value[0][1] - 0.8946847162442282) < 1e-15
    s = sequential_inhibition(0.1, (0, 2, 0, 2), max_failures=3000, seed=2)
    assert s.extra["saturated"] and min(math.dist(a, b) for a in s.value for b in s.value if a is not b) >= 0.1
    assert 0.45 < s.extra["coverage"] < 0.56
