"""mcspatial: estimators recomputed from the Philox draws, exact cases and statistical sanity."""

import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.mcspatial import mc_control_variate, mc_convergence, mc_integrate, mc_stratified, mc_zonal


def test_plain_integration_recomputed():
    r = mc_integrate(lambda x, y: x * y, bounds=[(0, 2), (1, 3)], n=500, seed=3)
    u = [float(v) for v in random_uniform(1000, seed=3, stream=0)]
    vals = [(2 * u[2 * i]) * (1 + 2 * u[2 * i + 1]) for i in range(500)]
    m = sum(vals) / 500
    assert r.mean == pytest.approx(m, abs=1e-12) and r.integral == pytest.approx(4 * m, abs=1e-12)
    assert r.se_mean == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in vals) / 499 / 500), abs=1e-12)
    assert abs(r.integral - 8.0) < 4 * r.se_integral  # exact integral of x y over [0,2]x[1,3] is 8


def test_antithetic_lhs_polygon():
    lin = mc_integrate(lambda x: 3 * x + 1, bounds=[(0, 1)], n=100, method="antithetic")
    assert lin.mean == pytest.approx(2.5, abs=1e-12) and lin.se_mean == pytest.approx(0.0, abs=1e-12)
    lhs = mc_integrate(lambda x: x, bounds=[(0, 1)], n=1000, method="lhs")
    assert abs(lhs.mean - 0.5) < 1e-3  # each of 1000 strata sampled once
    tri = mc_integrate(lambda x, y: 1.0, polygon=[(0, 0), (2, 0), (0, 2)], n=4000)
    assert tri.area == 2.0 and tri.integral == 2.0 and 1500 < tri.n_used < 2500


def test_stratified_zonal_control_convergence():
    sq = [[(0, 0), (0.5, 0), (0.5, 1), (0, 1)], [(0.5, 0), (1, 0), (1, 1), (0.5, 1)]]
    r = mc_stratified(lambda x, y: x, sq, n=400, allocation="neyman", pilot_sd=[1.0, 3.0])
    assert r.n_h == [100, 300] and abs(r.mean - 0.5) < 5 * r.se
    z = mc_zonal(lambda x, y: y < x, [(0, 1), (0, 1)], value=lambda x, y: x, n=20000)
    assert abs(z.area - 0.5) < 4 * z.se_area and abs(z.zone_mean - 2 / 3) < 4 * z.zone_mean_se
    y = [1.0, 2.0, 3.0, 5.0, 4.0]
    g = [1.1, 1.9, 3.2, 4.8, 4.1]
    cv = mc_control_variate(y, g, 3.0)
    my, mg = 3.0, sum(g) / 5
    b = sum((a - my) * (c - mg) for a, c in zip(y, g)) / sum((c - mg) ** 2 for c in g)
    assert cv.b == pytest.approx(b) and cv.estimate == pytest.approx(my - b * (mg - 3.0))
    c = mc_convergence([float(k % 5) for k in range(60)], batches=6)  # each batch holds two full periods
    assert c.running_mean[-1] == pytest.approx(2.0) and c.batch_se == pytest.approx(0.0, abs=1e-12)
