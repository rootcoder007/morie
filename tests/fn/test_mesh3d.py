"""Tests for mesh3d: 3-D Delaunay and Voronoi, Ruppert refinement."""

import math

from morie.fn.mesh3d import cheatsheet, delaunay_3d, ruppert_refine, voronoi_3d

P = [[math.sin(i * 1.3 + 0.2) * 2, math.cos(i * 2.7) * 1.5, math.sin(i * 0.37) + 0.1 * i] for i in range(30)]


def _d2(a, b):
    return sum((u - v) ** 2 for u, v in zip(a, b))


def _tet_volume(a, b, c, d):
    u, v, w = ([x[k] - a[k] for k in range(3)] for x in (b, c, d))
    return (
        abs(
            u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0])
        )
        / 6
    )


def test_delaunay_empty_circumsphere():
    r = delaunay_3d(P)
    for t, c in zip(r.tetrahedra, r.circumcenters):
        r2 = _d2(c, P[t[0]])
        assert all(abs(_d2(c, P[q]) - r2) <= 1e-9 * r2 for q in t)
        assert all(_d2(c, P[q]) >= r2 * (1 - 1e-9) for q in range(len(P)))


def test_delaunay_fills_the_unit_cube():
    cube = [[x, y, z + 0.01 * x * y] for x in (0, 1) for y in (0, 1) for z in (0, 1)] + [[0.5, 0.4, 0.45]]
    r = delaunay_3d(cube)
    total = sum(_tet_volume(*[cube[q] for q in t]) for t in r.tetrahedra)
    # hull: unit cube plus the pyramid of height 0.01 on top (0.01 / 3) minus the corner tetrahedron below (0.01 / 6)
    assert abs(total - (1.0 + 0.01 / 6)) <= 1e-12


def test_voronoi_vertices_are_nearest_and_volumes():
    v = voronoi_3d(P)
    for i in range(len(P)):
        for c in v.vertices[i]:
            assert _d2(c, P[i]) <= min(_d2(c, p) for p in P) * (1 + 1e-9)
    assert any(v.bounded) and not all(v.bounded)
    assert all((b and 0 < x < math.inf) or (not b and x == math.inf) for b, x in zip(v.bounded, v.volumes))
    lat = [[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
    lat = [[a + 1e-7 * math.sin(7 * k + j) for j, a in enumerate(p)] for k, p in enumerate(lat)]
    w = voronoi_3d(lat)
    assert w.bounded == [k == 13 for k in range(27)]
    assert abs(w.volumes[13] - 1.0) <= 1e-5


def _area(t, pts):
    a, b, c = (pts[q] for q in t)
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2


def test_ruppert_quality_and_area():
    Q = [[4.0 * ((i * 0.618) % 1), 2.0 * ((i * 0.414 + 0.3) % 1)] for i in range(9)]
    for ang in (15.0, 20.0):
        r = ruppert_refine(Q, min_angle=ang)
        assert r.min_angle >= ang
        assert r.points[: len(Q)] == Q
        for t in r.triangles:
            a, b, c = (r.points[q] for q in t)
            dd = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
            ux = sum((p[0] ** 2 + p[1] ** 2) * (s[1] - u[1]) for p, s, u in ((a, b, c), (b, c, a), (c, a, b))) / dd
            uy = sum((p[0] ** 2 + p[1] ** 2) * (u[0] - s[0]) for p, s, u in ((a, b, c), (b, c, a), (c, a, b))) / dd
            r2 = (a[0] - ux) ** 2 + (a[1] - uy) ** 2
            assert all((p[0] - ux) ** 2 + (p[1] - uy) ** 2 >= r2 * (1 - 1e-9) for p in r.points)
        hull = [r.points[s[0]] for s in r.segments]
        ha = abs(sum(hull[k][0] * hull[k - 1][1] - hull[k - 1][0] * hull[k][1] for k in range(len(hull)))) / 2
        assert abs(sum(_area(t, r.points) for t in r.triangles) - ha) <= 1e-9 * ha


def test_ruppert_max_points_caps():
    r = ruppert_refine([(0, 0), (10, 0), (10, 0.1), (0, 0.1)], min_angle=20, max_points=12)
    assert len(r.points) <= 12


def test_cheatsheet():
    assert "delaunay_3d" in cheatsheet()
