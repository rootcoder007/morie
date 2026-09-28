"""rfields: every transform recomputed from the underlying chol_sim draws; distributional sanity on large n."""

import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn._rrng_core import pnorm, qgamma, qnorm, qpois
from morie.fn.rfields import anisotropic_coords, max_stable_field, transformed_field
from morie.fn.zschl import chol_sim

P = [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (1.5, 1.5)]
M = {"model": "Exp", "psill": 4.0, "range": 1.2, "nugget": 1.0}
U = {"model": "Exp", "psill": 0.8, "range": 1.2, "nugget": 0.2}


def F(x):
    return [v for r in x for v in r]


def Z(seed, nsim=2, model=U):
    return chol_sim(P, model, nsim=nsim, seed=seed)["simulations"]


def test_pointwise_transforms_match_the_gaussian_draws():
    z = Z(5)
    assert F(transformed_field(P, M, "lognormal", seed=5, nsim=2, mean=1.0, sd=0.5).field) == pytest.approx(
        F([[math.exp(1 + 0.5 * v) for v in r] for r in z])
    )
    assert transformed_field(P, M, "binary", seed=5, nsim=2, threshold=0.3).field == [
        [1.0 if v > 0.3 else 0.0 for v in r] for r in z
    ]
    g = transformed_field(P, M, "gamma", seed=5, nsim=2, shape=3.0, rate=2.0).field
    assert F(g) == pytest.approx(F([[float(qgamma(float(pnorm(v)), 3.0, 2.0)) for v in r] for r in z]))
    c = transformed_field(P, M, "categorical", seed=5, nsim=2, proportions=[0.2, 0.5, 0.3]).field
    cuts = [float(qnorm(0.2)), float(qnorm(0.7))]
    assert c == [[float(sum(v > t for t in cuts)) for v in r] for r in z]
    ci = transformed_field(P, M, "cox_intensity", seed=5, nsim=2, scale=10.0).field
    assert F(ci) == pytest.approx(F([[10 * math.exp(v) for v in r] for r in z]))
    po = transformed_field(P, M, "poisson", seed=5, nsim=2, scale=10.0).field
    for s in range(2):
        u = random_uniform(4, seed=5, stream=500 + s)
        assert po[s] == [float(qpois(float(a), 10 * math.exp(v))) for a, v in zip(u, z[s])]
    with pytest.raises(ValueError):
        transformed_field(P, M, "categorical", proportions=[0.5, 0.4])
    with pytest.raises(ValueError):
        transformed_field(P, M, "bogus")


def test_chi2_t_mixture_white():
    z0, z1, z2, z3 = Z(7), Z(7 + 7919), Z(7 + 2 * 7919), Z(7 + 3 * 7919)
    ch = transformed_field(P, M, "chi2", seed=7, nsim=2, df=3).field
    assert F(ch) == pytest.approx(
        F([[z0[s][i] ** 2 + z1[s][i] ** 2 + z2[s][i] ** 2 for i in range(4)] for s in range(2)])
    )
    t = transformed_field(P, M, "student_t", seed=7, nsim=2, df=3).field
    want = [
        [z0[s][i] / math.sqrt((z1[s][i] ** 2 + z2[s][i] ** 2 + z3[s][i] ** 2) / 3) for i in range(4)] for s in range(2)
    ]
    assert F(t) == pytest.approx(F(want))
    G = {"model": "Gau", "psill": 2.0, "range": 1.0}
    mx = transformed_field(P, M, "mixture", seed=7, nsim=2, weights=[0.25, 0.75], models=[M, G]).field
    zg = chol_sim(P, {"model": "Gau", "psill": 1.0, "range": 1.0}, nsim=2, seed=7 + 7919)["simulations"]
    assert F(mx) == pytest.approx(
        F([[0.5 * z0[s][i] + math.sqrt(0.75) * zg[s][i] for i in range(4)] for s in range(2)])
    )
    w = transformed_field(P, M, "white", seed=2, nsim=300, mean=1.0, sd=2.0).field
    flat = [v for r in w for v in r]
    m = sum(flat) / len(flat)
    assert abs(m - 1) < 0.25 and abs(math.sqrt(sum((v - m) ** 2 for v in flat) / (len(flat) - 1)) - 2) < 0.2


def test_max_stable_and_anisotropy():
    r = max_stable_field(P, M, n_fields=50, seed=4)
    W = chol_sim(P, U, nsim=50, seed=4)["simulations"]
    u = random_uniform(50, seed=4, stream=2000)
    g, want = 0.0, [0.0] * 4
    for i in range(50):
        g -= math.log(float(u[i]))
        for k in range(4):
            want[k] = max(want[k], math.sqrt(2 * math.pi) / g * max(0.0, W[i][k]))
    assert r.field == pytest.approx(want, abs=1e-14)
    got = anisotropic_coords([(1.0, 2.0)], 30.0, 0.25)[0]
    a = math.radians(30)
    assert got == pytest.approx(((math.cos(a) - 2 * math.sin(a)) / 0.25, math.sin(a) + 2 * math.cos(a)))
    # distances along the major axis are unchanged
    p, q = anisotropic_coords([(0.0, 0.0), (math.sin(a), math.cos(a))], 30.0, 0.25)
    assert math.dist(p, q) == pytest.approx(1.0)
