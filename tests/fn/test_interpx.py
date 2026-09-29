"""interpx: exactness properties of the spline, barrier geometry and the potential."""

import math

import pytest

from morie.fn.interpx import barrier_idw, bicubic_spline, stewart_potential


def test_bicubic_reproduces_bilinear_and_nodes():
    x, y = [0.0, 0.5, 1.7, 2.0, 3.1], [-1.0, 0.2, 0.9, 2.5]
    f = lambda a, b: 1.5 - 0.7 * a + 2.2 * b + 0.4 * a * b  # noqa: E731
    z = [[f(a, b) for a in x] for b in y]
    pts = [(0.3, -0.4), (2.6, 2.1), (-0.5, 0.0), (3.5, 3.0)]
    got = bicubic_spline(x, y, z, [p[0] for p in pts], [p[1] for p in pts])
    assert got == pytest.approx([f(a, b) for a, b in pts], rel=1e-12, abs=1e-12)
    zz = [[math.sin(a + b) for a in x] for b in y]
    assert bicubic_spline(x, y, zz, [1.7], [0.9]) == pytest.approx([math.sin(2.6)], rel=1e-14)


def test_barrier_idw():
    bar = [((1.0, -5.0), (1.0, 5.0))]
    pts, vals = [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]], [1.0, 5.0, 2.0]
    vis = barrier_idw(pts, vals, [[0.5, 1.0]], bar)[0]
    d = [math.dist([0.5, 1.0], p) for p in pts[::2]]
    assert vis == pytest.approx((1 / d[0] ** 2 * 1 + 1 / d[1] ** 2 * 2) / (1 / d[0] ** 2 + 1 / d[1] ** 2), rel=1e-14)
    short = [((1.0, -1.0), (1.0, 1.0))]
    path = barrier_idw([[0.0, 0.0], [2.0, 0.0]], [0.0, 10.0], [[0.5, 0.0]], short, power=1, method="path")[0]
    d0 = 0.5
    d1 = math.dist([0.5, 0.0], [1.0, 1.0]) + math.dist([1.0, 1.0], [2.0, 0.0])
    assert path == pytest.approx((10 / d1) / (1 / d0 + 1 / d1), rel=1e-13)
    assert barrier_idw([[0, 0]], [3.0], [[0.0, 0.0]], short) == [3.0]


def test_stewart_potential():
    P, M = [[0, 0], [1, 0], [0, 2]], [10.0, 20.0, 5.0]
    v = stewart_potential(P, M, 2.0, self_distance=0.5)
    assert v[0] == pytest.approx(10 / 0.25 + 20 / 1 + 5 / 4, rel=1e-15)
