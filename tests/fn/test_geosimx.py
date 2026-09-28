import math

from morie.fn._rng import random_normal
from morie.fn.geosimx import (
    annealing_simulation,
    deformation_field_simulate,
    dla_aggregate,
    moving_least_squares,
    turning_bands_spherical,
)
from morie.fn.krgsys import kriging_covariance


def test_spherical_turning_bands_covariance():
    a, sill = 1.5, 2.0
    pts = [(0, 0, 0), (0.5, 0, 0), (0, 1.0, 0)]
    prods = {0: [], 1: [], 2: []}
    for s in range(300):
        f = turning_bands_spherical(pts, sill=sill, range_=a, n_bands=24, seed=s).field
        prods[0].append(f[0] * f[0])
        prods[1].append(f[0] * f[1])
        prods[2].append(f[0] * f[2])
    for k, h in ((0, 0.0), (1, 0.5), (2, 1.0)):
        x = h / a
        target = sill * (1 - 1.5 * x + 0.5 * x**3)
        v = prods[k]
        m = sum(v) / len(v)
        se = math.sqrt(sum((t - m) ** 2 for t in v) / (len(v) - 1) / len(v))
        assert abs(m - target) / se < 4


def test_dla_growth():
    r = dla_aggregate(120, seed=3)
    for k in range(1, len(r.sites)):
        x, y = r.sites[k]
        assert any(abs(x - a) + abs(y - b) == 1 for a, b in r.sites[:k])
    assert 1.2 < r.fractal_dimension < 2.2


def test_mls_reproduces_polynomials():
    P = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5), (2, 1), (1.5, 2), (0.2, 1.7)]
    lin = [1 + 2 * x - y for x, y in P]
    quad = [x * x - y + 0.3 * x * y for x, y in P]
    t = (0.3, 0.7)
    assert abs(moving_least_squares(P, lin, [t], 0.7).prediction[0] - (1 + 0.6 - 0.7)) < 1e-12
    assert abs(moving_least_squares(P, quad, [t], 0.9, degree=2).prediction[0] - (0.09 - 0.7 + 0.063)) < 1e-12


def test_deformation_covariance_and_draws():
    m = {"model": "Exp", "psill": 1.0, "range": 1.0}
    G = [(0, 0), (2, 0), (0, 0.5)]
    r = deformation_field_simulate([(0, 0), (1, 0), (0, 1)], G, m, nsim=2, seed=3)
    assert (
        abs(r.covariance[0][1] - math.exp(-2.0)) < 1e-15
        and abs(r.covariance[1][2] - math.exp(-math.hypot(2, 0.5))) < 1e-15
    )
    e = [float(v) for v in random_normal(3, seed=3, stream=0)]
    assert abs(r.realisations[0][0] - math.sqrt(1 + 1e-12) * e[0]) < 1e-12


def test_annealing_preserves_histogram():
    vals = [float((i * 7) % 11) for i in range(42)]
    m = {"model": "Exp", "psill": 1.0, "range": 3.0}
    r = annealing_simulation(7, 6, vals, m, [(1, 0), (0, 1)], n_iter=300, seed=2)
    assert sorted(v for row in r.grid for v in row) == sorted(vals)
    g = [v for row in r.grid for v in row]
    obj = 0.0
    for (dx, dy), tg in zip([(1, 0), (0, 1)], r.target):
        pr = [(y * 7 + x, (y + dy) * 7 + x + dx) for y in range(6) for x in range(7) if x + dx < 7 and y + dy < 6]
        gam = sum((g[a] - g[b]) ** 2 for a, b in pr) / (2 * len(pr))
        obj += ((gam - tg) / tg) ** 2
    assert abs(obj - r.objective[-1]) < 1e-9
    assert abs(r.target[0] - (1 - kriging_covariance(1.0, m))) < 1e-15
