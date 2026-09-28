import math

from morie.fn._qpcore import ssum
from morie.fn._rng import random_uniform
from morie.fn.hullgeo import alpha_shape, convex_hull_vertices, delaunay, hull_metrics, triangle_quality

U = [float(v) for v in random_uniform(80, seed=17, stream=0)]
PTS = [(U[i], U[40 + i]) for i in range(40)]


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def test_convex_hull_contains_all_points():
    H = convex_hull_vertices(PTS)
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]
        assert all(_cross(a, b, p) >= -1e-15 for p in PTS)
    assert set(H) <= set(PTS)
    assert convex_hull_vertices([(0, 0), (1, 1), (2, 0), (1, 0.5), (0, 2), (2, 2)]) == [
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (0.0, 2.0),
    ]


def test_hull_metrics_rectangles_and_l_shape():
    r = hull_metrics([(0, 0), (4, 0), (4, 1), (0, 1)])
    assert (r.solidity, r.elongation) == (1.0, 0.75)
    assert abs(r.roundness - 4 * math.pi * 4 / 100) < 1e-15
    assert abs(r.fractal_dimension - 2 * math.log(10 / 4) / math.log(4)) < 1e-15
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    rot = [(x * c - y * s, x * s + y * c) for x, y in [(0, 0), (4, 0), (4, 1), (0, 1)]]
    q = hull_metrics(rot)
    assert abs(q.elongation - 0.75) < 1e-12 and abs(q.orientation - 30) < 1e-9
    L = hull_metrics([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    assert abs(L.solidity - 3 / 3.5) < 1e-15


def test_delaunay_empty_circle_and_euler():
    T = delaunay(PTS)
    h = len(convex_hull_vertices(PTS))
    assert len(T) == 2 * 40 - 2 - h
    for t in T:
        a, b, c = (PTS[i] for i in t)
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        ux = (
            (a[0] ** 2 + a[1] ** 2) * (b[1] - c[1])
            + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
            + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])
        ) / d
        uy = (
            (a[0] ** 2 + a[1] ** 2) * (c[0] - b[0])
            + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
            + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])
        ) / d
        r = math.dist((ux, uy), a)
        assert all(math.dist((ux, uy), p) >= r - 1e-12 for p in PTS)


def test_alpha_shape_limits():
    H = convex_hull_vertices(PTS)
    n = len(H)
    area = abs(0.5 * ssum(H[i][0] * H[(i + 1) % n][1] - H[(i + 1) % n][0] * H[i][1] for i in range(n)))
    big = alpha_shape(PTS, 1e6)
    assert abs(big.area - area) < 1e-12 and len(big.edges) == n
    small = alpha_shape(PTS, 0.12)
    assert small.area < big.area and len(small.triangles) < len(big.triangles)
    r = alpha_shape([(0, 0), (2, 0), (0, 2), (2.2, 2.1)], 10)
    assert (round(r.area, 12), len(r.edges)) == (4.3, 4)


def test_triangle_quality():
    r = triangle_quality([(0, 0), (1, 0), (0.5, 3**0.5 / 2)])
    assert abs(r.quality[0] - 1) < 1e-15 and abs(r.edge_ratio[0] - 1) < 1e-15
    q = triangle_quality([(0, 0), (2, 0), (0, 1)], [(0, 1, 2)])
    assert abs(q.quality[0] - 4 * math.sqrt(3) * 1 / 10) < 1e-15 and abs(q.edge_ratio[0] - math.sqrt(5)) < 1e-15
    t = triangle_quality(PTS)
    assert 0 < t.min_quality <= t.mean_quality <= 1 and len(t.quality) == len(delaunay(PTS))
