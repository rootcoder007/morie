"""Clipped Voronoi tessellation (deldir::deldir with digits = 15)."""

import math

from morie.fn._rng import random_uniform
from morie.fn.vorcel import voronoi_cells


def pattern():
    U = [float(u) for u in random_uniform(120, seed=21, stream=0)]
    return [[2 * U[2 * i], U[2 * i + 1]] for i in range(60)]


def test_matches_deldir_tiles():
    r = voronoi_cells(pattern(), (0, 2, 0, 1))
    # deldir(x, y, rw = c(0, 2, 0, 1), digits = 15)$summary$dir.area[1:5]
    ref = [0.040615682817, 0.06288264359, 0.064367398718, 0.025502522808, 0.038516530999]
    assert all(abs(a - b) < 1e-12 for a, b in zip(r.value[:5], ref))
    assert abs(sum(r.value) - 2.0) < 1e-13
    assert r.extra["neighbours"][:3] == [[11, 22, 23, 55], [22, 32, 44, 56], [5, 19, 32, 45, 47, 53, 56, 58]]
    assert len(r.extra["edges"]) == 147 and sum(r.extra["neighbour_counts"]) == 294
    assert all(abs(i - 1 / a) < 1e-12 for i, a in zip(r.extra["intensity"], r.value))
    assert abs(r.extra["entropy"] - 1.7469656003642566) < 1e-12


def test_exact_cases():
    r = voronoi_cells([[0.25, 0.5], [0.75, 0.5]], (0, 1, 0, 1))
    assert r.value == [0.5, 0.5] and r.extra["perimeters"] == [3.0, 3.0] and r.extra["entropy"] == 0.0
    assert r.extra["centroids"] == [[0.25, 0.5], [0.75, 0.5]] and r.extra["edges"] == [[0, 1, 1.0]]
    # convex polygon window: areas are exact rationals (spatstat's integer clipping gives 0.276249999550)
    t = voronoi_cells([[1, 0.3], [0.6, 0.2], [1.4, 0.4], [1.0, 1.0]], [[0, 0], [2, 0], [1, 1.5]])
    assert abs(t.value[0] - 0.27625) < 1e-15 and abs(t.value[1] - 0.3803125) < 1e-15
    assert math.isclose(t.value[2], 0.420120192308, rel_tol=1e-11) and math.isclose(
        t.value[3], 0.423317307692, rel_tol=1e-11
    )
    assert abs(sum(t.value) - 1.5) < 1e-14
