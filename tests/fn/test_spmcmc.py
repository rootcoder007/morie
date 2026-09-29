"""Tests for spmcmc: spatial MCMC samplers on the Philox stream."""

import math

from morie.fn._rng import random_normal, random_uniform
from morie.fn.spmcmc import car_poisson_target, hmc_spatial, mh_spatial, nuts_spatial, tempered_spatial

A = [[1 if abs(i - j) == 1 else 0 for j in range(5)] for i in range(5)]
LP, GR = car_poisson_target([3, 5, 2, 8, 4], [3.2, 4.1, 3.0, 5.5, 3.9], A, tau=2.0, rho=0.8)


def test_car_gradient_matches_finite_differences():
    x = [0.1, 0.2, -0.1, 0.0, 0.3, -0.2]
    g = GR(x)
    for k in range(6):
        up, dn = list(x), list(x)
        up[k] += 1e-6
        dn[k] -= 1e-6
        assert abs((LP(up) - LP(dn)) / 2e-6 - g[k]) <= 1e-6


def test_mh_reproduces_its_streams():
    r = mh_spatial(LP, [0.0] * 6, 3, step=0.2, seed=9)
    x, lp = [0.0] * 6, LP([0.0] * 6)
    for t in range(3):
        z = random_normal(6, seed=9, stream=2 * t)
        u = float(random_uniform(1, seed=9, stream=2 * t + 1)[0])
        prop = [a + 0.2 * float(b) for a, b in zip(x, z)]
        if math.log(u) < LP(prop) - lp:
            x, lp = prop, LP(prop)
        assert r.samples[t] == x


def test_hmc_and_nuts_sample_a_standard_normal():
    def lp(x):
        return -0.5 * x[0] ** 2

    def gr(x):
        return [-x[0]]

    h = hmc_spatial(lp, gr, [0.0], 800, eps=0.3, n_leapfrog=8, seed=1)
    s = [v[0] for v in h.samples]
    assert abs(sum(s) / 800) < 0.15 and abs(sum(v * v for v in s) / 800 - 1) < 0.25
    assert h.acceptance > 0.9
    nu = nuts_spatial(lp, gr, [0.0], 800, eps=0.3, max_depth=6, seed=2)
    s = [v[0] for v in nu.samples]
    assert abs(sum(s) / 800) < 0.15 and abs(sum(v * v for v in s) / 800 - 1) < 0.25


def test_tempering_bimodal_target_visits_both_modes():
    def lp(x):
        return math.log(math.exp(-0.5 * ((x[0] - 4) / 0.5) ** 2) + math.exp(-0.5 * ((x[0] + 4) / 0.5) ** 2))

    r = tempered_spatial(lp, [4.0], 1500, [1.0, 3.0, 9.0, 27.0], step=0.6, seed=3)
    s = [v[0] for v in r.samples]
    assert any(v > 2 for v in s) and any(v < -2 for v in s)
    assert all(0.0 < v <= 1.0 for v in r.swap_rate)
