# morie.fn -- function file (rootcoder007/morie)
"""Delaunay triangulation by Bowyer-Watson insertion, with triangle quality statistics."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._qpcore import ssum


def _circum(a, b, c):
    bx, by = b[0] - a[0], b[1] - a[1]
    cx, cy = c[0] - a[0], c[1] - a[1]
    d = 2.0 * (bx * cy - by * cx)
    if d == 0.0:
        return None, math.inf
    b2, c2 = bx * bx + by * by, cx * cx + cy * cy
    ux = (cy * b2 - by * c2) / d
    uy = (bx * c2 - cx * b2) / d
    return (a[0] + ux, a[1] + uy), ux * ux + uy * uy


def _hull_area(P):
    pts = sorted(P)
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    h = lower[:-1] + upper[:-1]
    return 0.5 * ssum(h[k][0] * h[(k + 1) % len(h)][1] - h[(k + 1) % len(h)][0] * h[k][1] for k in range(len(h)))


def delaunay_triangulation(points) -> DescriptiveResult:
    """Delaunay triangulation of planar points with per-triangle statistics.

    Bowyer (1981) and Watson (1981) incremental insertion: each point in
    turn removes the triangles whose circumcircle contains it and re-links
    the cavity boundary to it. Coordinates are centred and scaled before
    insertion (the circumcircle test is then free of large-offset
    cancellation) and the enclosing super-triangle is 1000 times the
    extent of the points. Per triangle: area, circumcentre, circumradius
    ``R``, the three angles, the shortest and longest edges, the
    radius-edge ratio ``R / l_min`` (Shewchuk 2002) and the normalised
    shape quality ``4 sqrt(3) A / sum l^2`` (1 for an equilateral
    triangle). The empty-circle property is checked directly (the number
    of points strictly inside some circumcircle is reported, 0 for a
    Delaunay triangulation), and the triangles' total area is compared
    with the convex hull's. Edges give the Delaunay neighbours and the
    binary spatial-weights matrix.

    :param points: (n, 2) distinct points, ``n >= 3``, not all collinear.
    :return: DescriptiveResult; ``value`` is the triangles (index triples,
        counter-clockwise); ``extra`` has ``edges``, ``neighbours``,
        ``adjacency``, ``areas``, ``circumcentres``, ``circumradii``,
        ``angles`` (degrees), ``min_angle``, ``radius_edge``, ``quality``,
        ``empty_circle_violations`` and ``hull_area_gap``.

    References
    ----------
    Bowyer, A. (1981). Computing Dirichlet tessellations. Computer Journal
    24, 162-166.

    Watson, D. F. (1981). Computing the n-dimensional Delaunay tessellation
    with application to Voronoi polytopes. Computer Journal 24, 167-172.

    Shewchuk, J. R. (2002). Delaunay refinement algorithms for triangular
    mesh generation. Computational Geometry 22, 21-74.

    Examples
    --------
    >>> r = delaunay_triangulation([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.4]])
    >>> len(r.value), r.extra["empty_circle_violations"], sorted(r.extra["neighbours"][4])
    (4, 0, [0, 1, 2, 3])
    """
    P = [[float(v) for v in p] for p in points]
    n = len(P)
    if n < 3:
        raise ValueError("need at least three points")
    mx = ssum(p[0] for p in P) / n
    my = ssum(p[1] for p in P) / n
    sc = max(max(abs(p[0] - mx), abs(p[1] - my)) for p in P) or 1.0
    Q = [((p[0] - mx) / sc, (p[1] - my) / sc) for p in P]
    allp = Q + [(-3000.0, -3000.0), (3000.0, -3000.0), (0.0, 3000.0)]
    tris = [(n, n + 1, n + 2)]
    cache = {tris[0]: _circum(*(allp[k] for k in tris[0]))}
    for pi in range(n):
        px, py = allp[pi]
        bad = []
        for t in tris:
            c, r2 = cache[t]
            if c is not None and (px - c[0]) ** 2 + (py - c[1]) ** 2 < r2 * (1.0 + 1e-12):
                bad.append(t)
        count = {}
        for t in bad:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key = (min(e), max(e))
                count[key] = count.get(key, 0) + 1
        badset = set(bad)
        tris = [t for t in tris if t not in badset]
        for t in bad:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                if count[(min(e), max(e))] == 1:
                    nt = (e[0], e[1], pi)
                    tris.append(nt)
                    cache[nt] = _circum(*(allp[k] for k in nt))
    out = []
    for t in tris:
        if max(t) >= n:
            continue
        a, b, c = (P[k] for k in t)
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
            t = (t[0], t[2], t[1])
        out.append(list(t))
    out.sort()
    areas, cent, rad, angles, mins, redge, qual = [], [], [], [], [], [], []
    for t in out:
        a, b, c = (P[k] for k in t)
        A = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        cc, r2 = _circum(a, b, c)
        la, lb, lc = math.dist(b, c), math.dist(a, c), math.dist(a, b)
        ang = [
            math.degrees(math.acos(max(-1.0, min(1.0, (lb * lb + lc * lc - la * la) / (2 * lb * lc))))),
            math.degrees(math.acos(max(-1.0, min(1.0, (la * la + lc * lc - lb * lb) / (2 * la * lc))))),
        ]
        ang.append(180.0 - ang[0] - ang[1])
        areas.append(A)
        cent.append(list(cc))
        rad.append(math.sqrt(r2))
        angles.append(ang)
        mins.append(min(ang))
        redge.append(math.sqrt(r2) / min(la, lb, lc))
        qual.append(4.0 * math.sqrt(3.0) * A / (la * la + lb * lb + lc * lc))
    edges = sorted({(min(e), max(e)) for t in out for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))})
    nb = [[] for _ in range(n)]
    for i, j in edges:
        nb[i].append(j)
        nb[j].append(i)
    viol = 0
    for t, cc, r in zip(out, cent, rad):
        for k in range(n):
            if k not in t and math.dist(P[k], cc) < r * (1 - 1e-10):
                viol += 1
    return DescriptiveResult(
        name="delaunay_triangulation",
        value=out,
        extra={
            "edges": [list(e) for e in edges],
            "neighbours": [sorted(v) for v in nb],
            "adjacency": [[1 if j in set(v) else 0 for j in range(n)] for v in nb],
            "areas": areas,
            "circumcentres": cent,
            "circumradii": rad,
            "angles": angles,
            "min_angle": mins,
            "radius_edge": redge,
            "quality": qual,
            "empty_circle_violations": viol,
            "hull_area_gap": _hull_area([tuple(p) for p in P]) - ssum(areas),
        },
    )


deltri = delaunay_triangulation


def cheatsheet() -> str:
    return "delaunay_triangulation(points) -> Bowyer-Watson Delaunay triangles with quality statistics"
