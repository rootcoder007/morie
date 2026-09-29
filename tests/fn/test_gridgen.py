"""gridgen: geometric and design properties."""

import math

import pytest

from morie.fn.gridgen import regular_grid, space_time_sample, systematic_sample, transfinite_grid, triangular_grid


def test_square_and_hex_grids():
    g = regular_grid((1.0, 2.0, 4.5, 3.9), 0.5)
    assert len(g) == 7 * 4 and g[0] == [1.25, 2.25] and g[-1] == [4.25, 3.75]
    h = regular_grid((0.0, 0.0, 3.0, 3.0), 1.0, "hexagon")
    nn = [min(math.dist(p, q) for q in h if q is not p) for p in h]
    assert min(nn) == pytest.approx(1.0, rel=1e-12) and max(nn) == pytest.approx(1.0, rel=1e-12)


def test_triangles_are_equilateral():
    t = triangular_grid((0, 0, 4, 3), 1.2)
    for a, b, c in t.triangles:
        V = t.vertices
        for p, q in ((a, b), (b, c), (c, a)):
            assert math.dist(V[p], V[q]) == pytest.approx(1.2, rel=1e-12)


def test_coons_patch_reproduces_boundaries_and_bilinear_map():
    B = [[i / 3, 0.2 * i / 3] for i in range(4)]
    T = [[i / 3, 1 + 0.5 * i / 3] for i in range(4)]
    L = [[0, j / 2] for j in range(3)]
    R = [[1, 0.2 + j / 2 * 1.3] for j in range(3)]
    g = transfinite_grid(B, T, L, R)
    flat = lambda rows: [c for r in rows for c in r]  # noqa: E731
    assert flat(g[0]) == pytest.approx(flat(B)) and flat(g[-1]) == pytest.approx(flat(T))
    assert flat([row[0] for row in g]) == pytest.approx(flat(L))
    assert flat([row[-1] for row in g]) == pytest.approx(flat(R))
    u, v = 1 / 3, 1 / 2
    bil = [
        (1 - v) * (1 - u) * 0 + (1 - v) * u * 1 + v * (1 - u) * 0 + v * u * 1,
        (1 - v) * (1 - u) * 0 + (1 - v) * u * 0.2 + v * (1 - u) * 1 + v * u * 1.5,
    ]
    assert g[1][1] == pytest.approx(bil, rel=1e-12)


def test_systematic_sample_and_designs():
    s = systematic_sample((0, 0, 10, 6), 2.0, seed=3)
    xs = sorted({round(p[0], 12) for p in s})
    assert all(abs((b - a) - 2.0) < 1e-12 for a, b in zip(xs, xs[1:]))
    tri = systematic_sample((0, 0, 10, 10), 1.0, seed=1, polygon=[[0, 0], [10, 0], [0, 10]])
    assert all(p[0] + p[1] < 10 for p in tri)
    syn = space_time_sample(50, 5, 3, "synchronous", seed=1)
    assert all(len(s) == 5 and len(set(s)) == 5 for s in syn) and syn[0] != syn[1]
    ss = space_time_sample(50, 6, 3, "static_synchronous", n_static=4, seed=1)
    common = set(ss[0]) & set(ss[1]) & set(ss[2])
    assert len(common) >= 4 and all(len(s) == 6 for s in ss)
    rot = space_time_sample(50, 6, 4, "rotating", rotation=3, seed=1)
    assert all(len(set(rot[t]) & set(rot[t + 1])) == 4 for t in range(3))
