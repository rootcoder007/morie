import itertools
import math

from morie.fn._rng import random_uniform
from morie.fn.spsampling import (
    adaptive_cluster_sample,
    cell_declustering_weights,
    hexagonal_grid_sample,
    map_quality_indices,
    quadtree_grid,
    random_spatial_sample,
    spatial_thinning,
    voronoi_declustering_weights,
)

POLY = [(0, 0), (10, 0), (12, 6), (4, 9), (-1, 5)]


def test_random_sample_uses_rejection_order():
    pts = random_spatial_sample(3, [(0, 0), (1, 0), (1, 1), (0, 1)], seed=3)
    u = [float(v) for v in random_uniform(2 * 64, seed=3, stream=0)]
    assert pts[:2] == [(u[0], u[1]), (u[2], u[3])]


def test_hexagonal_lattice_spacing():
    h = hexagonal_grid_sample(POLY, 1.7, offset=(0.3, 0.6))
    for p in h:
        d = min(math.dist(p, q) for q in h if q != p)
        assert abs(d - 1.7) < 1e-9


def test_acs_estimators_unbiased_by_enumeration():
    Y = [[0, 0, 0, 0], [0, 5, 7, 0], [0, 3, 0, 1]]
    mu = sum(map(sum, Y)) / 12
    hh = ht = 0.0
    combos = list(itertools.combinations([(r, c) for r in range(3) for c in range(4)], 2))
    for s in combos:
        r = adaptive_cluster_sample(Y, list(s), threshold=1)
        hh += r.mean_hh
        ht += r.mean_ht
    assert abs(hh / len(combos) - mu) < 1e-12 and abs(ht / len(combos) - mu) < 1e-12


def test_quadtree_partition():
    u = [float(v) for v in random_uniform(100, seed=2)]
    pts = [(u[2 * i] ** 2, u[2 * i + 1]) for i in range(50)]
    r = quadtree_grid(pts, (0, 0, 1, 1), capacity=4)
    assert sum(c[5] for c in r.cells) == 50
    assert abs(sum((c[2] - c[0]) * (c[3] - c[1]) for c in r.cells) - 1) < 1e-12
    assert all(c[5] <= 4 or c[4] == 8 for c in r.cells)


def test_declustering_weights():
    grid = [(x + 0.5, y + 0.5) for x in range(3) for y in range(3)]
    v = voronoi_declustering_weights(grid, (0, 0, 3, 3))
    assert all(abs(w - 1 / 9) < 1e-12 for w in v.weights)
    u = [float(v) for v in random_uniform(60, seed=4)]
    pts = [(3 * u[2 * i], 3 * u[2 * i + 1]) for i in range(30)]
    assert abs(sum(voronoi_declustering_weights(pts, (0, 0, 3, 3)).areas) - 9) < 1e-9
    c = cell_declustering_weights(pts, 1.0)
    assert abs(sum(c.weights) - 1) < 1e-12


def test_thinning_is_maximal():
    u = [float(v) for v in random_uniform(80, seed=5)]
    pts = [(5 * u[2 * i], 5 * u[2 * i + 1]) for i in range(40)]
    kept = [k - 1 for k in spatial_thinning(pts, 1.0, reps=5).kept]
    assert all(math.dist(pts[a], pts[b]) >= 1.0 for a, b in itertools.combinations(kept, 2))
    assert all(any(math.dist(pts[i], pts[k]) < 1.0 for k in kept) for i in range(40) if i not in kept)


def test_map_quality():
    z, h = [1.0, 2.5, 3.1, 4.0, 2.2], [1.4, 2.1, 3.3, 3.5, 2.0]
    r = map_quality_indices(z, h)
    e = [b - a for a, b in zip(z, h)]
    zb, hb = sum(z) / 5, sum(h) / 5
    cov = sum((a - zb) * (b - hb) for a, b in zip(z, h))
    r2 = cov**2 / (sum((a - zb) ** 2 for a in z) * sum((b - hb) ** 2 for b in h))
    assert abs(r.ME - sum(e) / 5) < 1e-15 and abs(r.r2 - r2) < 1e-14
    assert abs(r.MEC - (1 - sum(v * v for v in e) / sum((a - zb) ** 2 for a in z))) < 1e-14
    w = map_quality_indices(z, h, pi=[0.125] * 5, N=40)
    assert abs(w.MSE - r.MSE) < 1e-14  # SRS: pi = n / N gives the sample mean
