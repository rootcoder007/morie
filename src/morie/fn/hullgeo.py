# morie.fn -- function file (rootcoder007/morie)
"""Convex-hull shape metrics (solidity, elongation, fractal dimension, roundness) and alpha shapes
(concave hulls) from the Delaunay triangulation."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = ["convex_hull", "hull_metrics", "delaunay", "triangle_quality", "alpha_shape"]


def _pts(P):
    return [(float(a), float(b)) for a, b in (P.tolist() if hasattr(P, "tolist") else P)]


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points) -> list:
    r"""Convex hull vertices, counter-clockwise from the lexicographically smallest point (Andrew's monotone chain); collinear points dropped.

    References
    ----------
    Andrew, A. M. (1979). Another efficient algorithm for convex hulls in two
    dimensions. *Information Processing Letters*, 9(5), 216-219.

    Examples
    --------
    >>> convex_hull([(0, 0), (1, 1), (2, 0), (1, 0.5), (0, 2), (2, 2)])
    [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    """
    S = sorted(set(_pts(points)))
    if len(S) <= 2:
        return S

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and _cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h

    lo, up = half(S), half(S[::-1])
    return lo[:-1] + up[:-1]


def _area(poly):
    n = len(poly)
    return 0.5 * ssum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))


def _perim(poly):
    n = len(poly)
    return ssum(math.dist(poly[i], poly[(i + 1) % n]) for i in range(n))


def hull_metrics(polygon) -> RichResult:
    r"""Shape metrics of a polygon (vertices in order) or point set through its convex hull.

    - ``solidity`` (area ratio): polygon area / convex hull area (1 for a
      convex polygon);
    - ``elongation``: ``1 - w / l`` of the minimum-area enclosing rectangle
      (rotating calipers over the hull edges; Freeman and Shapira 1975), with
      its ``width``, ``length`` and ``orientation`` (degrees of the long side);
    - ``fractal_dimension``: ``2 ln(P / 4) / ln(A)`` of the hull (the FRAGSTATS
      ``FRAC`` index; McGarigal and Marks 1995), 1 for a square of any size
      only asymptotically, so it is reported with ``fractal_dimension_polygon``;
    - ``roundness``: hull isoperimetric quotient ``4 pi A / P^2`` (1 for a
      circle) and ``compactness`` the same for the polygon.

    References
    ----------
    Freeman, H. and Shapira, R. (1975). Determining the minimum-area encasing
    rectangle for an arbitrary closed curve. *Communications of the ACM*,
    18(7), 409-413.
    McGarigal, K. and Marks, B. J. (1995). FRAGSTATS: spatial pattern analysis
    program for quantifying landscape structure. USDA Forest Service General
    Technical Report PNW-351.

    Examples
    --------
    >>> r = hull_metrics([(0, 0), (4, 0), (4, 1), (0, 1)])
    >>> r.solidity, r.elongation, round(r.roundness, 12)
    (1.0, 0.75, 0.502654824574)
    """
    poly = _pts(polygon)
    H = convex_hull(poly)
    ah, ph = abs(_area(H)), _perim(H)
    ap, pp = abs(_area(poly)), _perim(poly)
    best = None
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]
        L = math.dist(a, b)
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        s = [(p[0] - a[0]) * ux + (p[1] - a[1]) * uy for p in H]
        t = [-(p[0] - a[0]) * uy + (p[1] - a[1]) * ux for p in H]
        w1, w2 = max(s) - min(s), max(t) - min(t)
        if best is None or w1 * w2 < best[0] - 1e-12 * max(1.0, best[0]):
            best = (w1 * w2, w1, w2, math.degrees(math.atan2(uy, ux)))
    _, w1, w2, ang = best
    length, width = max(w1, w2), min(w1, w2)
    orient = ang if w1 >= w2 else ang + 90.0
    return RichResult(
        payload={
            "hull": H,
            "hull_area": ah,
            "hull_perimeter": ph,
            "area": ap,
            "perimeter": pp,
            "solidity": ap / ah if ah > 0 else math.nan,
            "elongation": 1 - width / length if length > 0 else 0.0,
            "width": width,
            "length": length,
            "orientation": orient % 180.0,
            "fractal_dimension": 2 * math.log(ph / 4) / math.log(ah) if ah > 0 and ah != 1 else math.nan,
            "fractal_dimension_polygon": 2 * math.log(pp / 4) / math.log(ap) if ap > 0 and ap != 1 else math.nan,
            "roundness": 4 * math.pi * ah / ph**2 if ph > 0 else math.nan,
            "compactness": 4 * math.pi * ap / pp**2 if pp > 0 else math.nan,
        }
    )


def _circum(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if d == 0:
        return None
    a2, b2, c2 = a[0] ** 2 + a[1] ** 2, b[0] ** 2 + b[1] ** 2, c[0] ** 2 + c[1] ** 2
    ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d
    uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d
    return ux, uy, math.dist((ux, uy), a)


def delaunay(points) -> list:
    r"""Delaunay triangulation (Bowyer-Watson), triangles as sorted index triples.

    Points are inserted in order into a super-triangle; triangles whose
    circumcircle strictly contains the new point are removed and the cavity
    re-triangulated (Bowyer 1981; Watson 1981). Assumes no four points
    cocircular (general position); duplicate points are not allowed.

    References
    ----------
    Bowyer, A. (1981). Computing Dirichlet tessellations. *The Computer
    Journal*, 24(2), 162-166.
    Watson, D. F. (1981). Computing the n-dimensional Delaunay tessellation
    with application to Voronoi polytopes. *The Computer Journal*, 24(2),
    167-172.

    Examples
    --------
    >>> delaunay([(0, 0), (2, 0), (0, 2), (2.2, 2.1)])
    [(0, 1, 2), (1, 2, 3)]
    """
    P = _pts(points)
    n = len(P)
    xs, ys = [p[0] for p in P], [p[1] for p in P]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    d = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) * 20
    V = P + [(cx - d, cy - d), (cx + d, cy - d), (cx, cy + d)]
    tris = {(n, n + 1, n + 2): _circum(V[n], V[n + 1], V[n + 2])}
    for i in range(n):
        p = V[i]
        bad = [t for t, c in tris.items() if c is not None and math.dist((c[0], c[1]), p) < c[2]]
        edges = {}
        for t in bad:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2])):
                e = tuple(sorted(e))
                edges[e] = edges.get(e, 0) + 1
            del tris[t]
        for (a, b), k in edges.items():
            if k == 1:
                t = tuple(sorted((a, b, i)))
                tris[t] = _circum(V[t[0]], V[t[1]], V[t[2]])
    return sorted(t for t in tris if max(t) < n)


def triangle_quality(points, triangles=None) -> RichResult:
    r"""Area/edge quality ``q = 4 sqrt(3) A / (a^2 + b^2 + c^2)`` of each triangle (1 equilateral, 0 degenerate) of a triangulation.

    Triangles default to :func:`delaunay` of ``points``. Also returns the
    areas, the longest-to-shortest edge ratios and the minimum and mean
    quality (Bank's shape measure; Field 2000).

    References
    ----------
    Field, D. A. (2000). Qualitative measures for initial meshes.
    *International Journal for Numerical Methods in Engineering*, 47(4),
    887-906.

    Examples
    --------
    >>> round(triangle_quality([(0, 0), (1, 0), (0.5, 3 ** 0.5 / 2)]).quality[0], 12)
    1.0
    """
    P = _pts(points)
    T = delaunay(P) if triangles is None else [tuple(int(v) for v in t) for t in triangles]
    q, area, ratio = [], [], []
    for t in T:
        a, b, c = P[t[0]], P[t[1]], P[t[2]]
        e = [math.dist(a, b), math.dist(b, c), math.dist(a, c)]
        A = abs(_area([a, b, c]))
        q.append(4 * math.sqrt(3) * A / ssum(v * v for v in e))
        area.append(A)
        ratio.append(max(e) / min(e))
    return RichResult(
        payload={
            "triangles": T,
            "quality": q,
            "area": area,
            "edge_ratio": ratio,
            "min_quality": min(q),
            "mean_quality": ssum(q) / len(q),
        }
    )


def alpha_shape(points, radius: float) -> RichResult:
    r"""Alpha shape (concave hull) of a point set: Delaunay triangles with circumradius at most ``radius``, and their boundary.

    Edelsbrunner, Kirkpatrick and Seidel's alpha complex restricted to
    triangles (the regularised alpha shape): as ``radius`` grows it becomes
    the convex hull, as it shrinks it hugs the points. Returns the kept
    ``triangles``, the ``area``, the boundary ``edges`` (sides of exactly one
    kept triangle) and the ``perimeter``.

    References
    ----------
    Edelsbrunner, H., Kirkpatrick, D. G. and Seidel, R. (1983). On the shape
    of a set of points in the plane. *IEEE Transactions on Information
    Theory*, 29(4), 551-559.

    Examples
    --------
    >>> r = alpha_shape([(0, 0), (2, 0), (0, 2), (2.2, 2.1)], 10)
    >>> round(r.area, 12), len(r.edges)
    (4.3, 4)
    """
    P = _pts(points)
    keep = []
    for t in delaunay(P):
        c = _circum(P[t[0]], P[t[1]], P[t[2]])
        if c is not None and c[2] <= radius:
            keep.append(t)
    cnt = {}
    for t in keep:
        for e in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2])):
            cnt[e] = cnt.get(e, 0) + 1
    edges = sorted(e for e, k in cnt.items() if k == 1)
    area = ssum(abs(_area([P[t[0]], P[t[1]], P[t[2]]])) for t in keep)
    return RichResult(
        payload={
            "triangles": keep,
            "area": area,
            "edges": edges,
            "perimeter": ssum(math.dist(P[a], P[b]) for a, b in edges),
        }
    )


def cheatsheet() -> str:
    return "convex_hull / hull_metrics / delaunay / triangle_quality / alpha_shape -> hull shape metrics and concave hulls."
