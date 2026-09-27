"""Bowyer-Watson Delaunay triangulation (deldir::triang.list, geometry::delaunayn)."""

import math

from morie.fn._rng import random_uniform
from morie.fn.deltri import delaunay_triangulation


def pattern():
    U = [float(u) for u in random_uniform(120, seed=21, stream=0)]
    return [[2 * U[2 * i], U[2 * i + 1]] for i in range(60)]


def test_analytic_triangles():
    r = delaunay_triangulation([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.4]])
    assert r.value == [[0, 1, 4], [1, 3, 4], [2, 0, 4], [3, 2, 4]]
    # circumcentre of (0,0), (1,0), (0.5,0.4) is (0.5, -0.1125): R^2 = 0.25 + 0.1125^2
    assert abs(r.extra["circumradii"][0] - math.sqrt(0.26265625)) < 1e-15
    c = r.extra["circumcentres"][0]
    assert abs(c[0] - 0.5) < 1e-15 and abs(c[1] + 0.1125) < 1e-15 and r.extra["empty_circle_violations"] == 0
    e = delaunay_triangulation([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]])
    assert abs(e.extra["quality"][0] - 1) < 1e-15 and abs(e.extra["circumradii"][0] - 1 / math.sqrt(3)) < 1e-15
    assert all(abs(a - 60) < 1e-12 for a in e.extra["angles"][0])


def test_matches_deldir_and_is_shift_invariant():
    P = pattern()
    r = delaunay_triangulation(P)
    # deldir(x, y, rw = c(0, 2, 0, 1))$ triangles: 108, identical index sets
    assert len(r.value) == 108 and r.extra["empty_circle_violations"] == 0 and abs(r.extra["hull_area_gap"]) < 1e-13
    s = delaunay_triangulation([[p[0] + 1e6, p[1] - 2e6] for p in P])
    assert sorted(sorted(t) for t in s.value) == sorted(sorted(t) for t in r.value)
    assert sum(len(v) for v in r.extra["neighbours"]) == 2 * len(r.extra["edges"])
    assert all(r.extra["adjacency"][i][j] == 1 for i, j in r.extra["edges"])
