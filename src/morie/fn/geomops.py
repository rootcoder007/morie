# morie.fn -- function file (rootcoder007/morie)
"""Vector and raster geoprocessing: polygon clip, erase, intersection and union (Greiner-Hormann), filled
contour bands of a gridded surface, elevation profiles along a path, and multiple-ring proximity bands."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = ["polygon_boolean", "polygon_area", "filled_contour_bands", "elevation_profile", "proximity_bands"]


def polygon_area(ring) -> float:
    r"""Signed shoelace area of a ring (positive when counter-clockwise).

    Examples
    --------
    >>> polygon_area([(0, 0), (2, 0), (2, 1), (0, 1)])
    2.0
    """
    s = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _inside(p, poly):
    x, y = p
    c = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            c = not c
    return c


def _seg_int(p1, p2, q1, q2):
    d = (p2[0] - p1[0]) * (q2[1] - q1[1]) - (p2[1] - p1[1]) * (q2[0] - q1[0])
    if d == 0:
        return None
    a = ((q1[0] - p1[0]) * (q2[1] - q1[1]) - (q1[1] - p1[1]) * (q2[0] - q1[0])) / d
    b = ((q1[0] - p1[0]) * (p2[1] - p1[1]) - (q1[1] - p1[1]) * (p2[0] - p1[0])) / d
    if 0 < a < 1 and 0 < b < 1:
        return a, b, (p1[0] + a * (p2[0] - p1[0]), p1[1] + a * (p2[1] - p1[1]))
    return None


def polygon_boolean(subject, clip, op: str = "intersection") -> RichResult:
    r"""Boolean operations on two simple polygons by the Greiner-Hormann algorithm.

    ``op`` is ``"intersection"`` (overlay / clip the subject to the clip
    polygon), ``"union"`` (dissolve) or ``"difference"`` (erase the clip
    polygon from the subject). Edge intersections are inserted into both
    vertex rings, flagged entry/exit by point-in-polygon tests (flags
    toggled for union, and for the subject in a difference), and the result
    rings are traced by walking forward on entries and backward on exits,
    switching polygons at every intersection. Without edge crossings the
    containment cases are resolved directly (a hole is returned clockwise).
    Returns the rings and the total (signed) area. General position is
    assumed (no vertex on the other polygon's boundary).

    References
    ----------
    Greiner, G. and Hormann, K. (1998). Efficient clipping of arbitrary
    polygons. *ACM Transactions on Graphics*, 17, 71-83.

    Examples
    --------
    >>> A = [(0, 0), (4, 0), (4, 3), (0, 3)]
    >>> B = [(2, 1), (6, 2), (5, 5)]
    >>> [round(polygon_boolean(A, B, op).area, 10) for op in ("intersection", "union", "difference")]
    [2.0, 16.5, 10.0]
    """
    S = [(float(a), float(b)) for a, b in subject]
    C = [(float(a), float(b)) for a, b in clip]
    if polygon_area(S) < 0:
        S = S[::-1]
    if polygon_area(C) < 0:
        C = C[::-1]
    if op not in ("intersection", "union", "difference"):
        raise ValueError("op must be intersection, union or difference")
    # node lists: dict(pt, inter, alpha, neighbor, entry, visited)
    sn = [{"pt": p, "inter": False} for p in S]
    cn = [{"pt": p, "inter": False} for p in C]
    sins = [[] for _ in S]
    cins = [[] for _ in C]
    for i in range(len(S)):
        for j in range(len(C)):
            r = _seg_int(S[i], S[(i + 1) % len(S)], C[j], C[(j + 1) % len(C)])
            if r is None:
                continue
            a, b, p = r
            ns = {"pt": p, "inter": True, "alpha": a, "visited": False}
            nc = {"pt": p, "inter": True, "alpha": b, "visited": False}
            ns["nb"], nc["nb"] = nc, ns
            sins[i].append(ns)
            cins[j].append(nc)

    def build(base, ins):
        out = []
        for k, v in enumerate(base):
            out.append(v)
            out.extend(sorted(ins[k], key=lambda d: d["alpha"]))
        return out

    SL, CL = build(sn, sins), build(cn, cins)
    for lst in (SL, CL):
        for k, v in enumerate(lst):
            v["next"] = lst[(k + 1) % len(lst)]
            v["prev"] = lst[k - 1]
    if not any(v["inter"] for v in SL):
        s_in_c, c_in_s = _inside(S[0], C), _inside(C[0], S)
        if op == "intersection":
            rings = [S] if s_in_c else [C] if c_in_s else []
        elif op == "union":
            rings = [C] if s_in_c else [S] if c_in_s else [S, C]
        else:
            rings = [] if s_in_c else [S, C[::-1]] if c_in_s else [S]
        return RichResult(payload={"rings": rings, "area": ssum(polygon_area(r) for r in rings)})
    for lst, other, flip in ((SL, C, op in ("union", "difference")), (CL, S, op == "union")):
        entry = not _inside(lst[0]["pt"], other)
        if flip:
            entry = not entry
        for v in lst:
            if v["inter"]:
                v["entry"] = entry
                entry = not entry
    rings = []
    for start in SL:
        if not start["inter"] or start["visited"]:
            continue
        ring = []
        cur = start
        while True:
            cur["visited"] = True
            cur["nb"]["visited"] = True
            ring.append(cur["pt"])
            if cur["entry"]:
                cur = cur["next"]
                while not cur["inter"]:
                    ring.append(cur["pt"])
                    cur = cur["next"]
            else:
                cur = cur["prev"]
                while not cur["inter"]:
                    ring.append(cur["pt"])
                    cur = cur["prev"]
            cur = cur["nb"]
            if cur["visited"]:
                break
        rings.append(ring)
    # traversal directions are not reliable orientations: make every ring counter-clockwise, then reverse
    # the rings nested an odd number of times in others (holes)
    rings = [r if polygon_area(r) > 0 else r[::-1] for r in rings]
    out = []
    for i, r in enumerate(rings):
        e = 0
        while e < len(r):
            p, q = r[e], r[(e + 1) % len(r)]
            t = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            if all(
                min(_seg_dist(t, o[k], o[(k + 1) % len(o)]) for k in range(len(o))) > 1e-9
                for j, o in enumerate(rings)
                if j != i
            ):
                break
            e += 1
        depth = sum(1 for j, o in enumerate(rings) if j != i and _inside(t, o))
        out.append(r[::-1] if depth % 2 else r)
    return RichResult(payload={"rings": out, "area": ssum(polygon_area(r) for r in out)})


def _clip_linear(poly, vals, level, keep_above):
    """Clip a convex polygon (with values at its vertices, linear inside) to v >= level (or v <= level)."""
    out, ov = [], []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        a, b = vals[i], vals[(i + 1) % n]
        ina = a >= level if keep_above else a <= level
        inb = b >= level if keep_above else b <= level
        if ina:
            out.append(p)
            ov.append(a)
        if ina != inb:
            t = (level - a) / (b - a)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            ov.append(level)
    return out, ov


def filled_contour_bands(x, y, z, levels) -> RichResult:
    r"""Filled contours (isobands) of a gridded surface and the area of each band.

    Every grid cell is split into two triangles on which the surface is
    linear; the part of a triangle with ``levels[k] <= z <= levels[k+1]`` is
    the triangle clipped by the two level lines, so band polygons and areas
    are exact for the piecewise-linear surface (no marching-squares saddle
    ambiguity). ``z[j][i]`` is the value at ``(x[i], y[j])``.

    References
    ----------
    Lorensen, W. E. and Cline, H. E. (1987). Marching cubes: a high resolution
    3D surface construction algorithm. *Computer Graphics*, 21(4), 163-169.
    Watson, D. F. (1992). *Contouring: A Guide to the Analysis and Display of
    Spatial Data*. Pergamon.

    Examples
    --------
    >>> r = filled_contour_bands([0, 1], [0, 1], [[0, 1], [1, 2]], [0, 1, 2])
    >>> r.areas
    [0.5, 0.5]
    """
    X, Y = [float(v) for v in x], [float(v) for v in y]
    Z = [[float(v) for v in row] for row in z]
    L = [float(v) for v in levels]
    nb = len(L) - 1
    areas = [0.0] * nb
    polys = [[] for _ in range(nb)]
    for j in range(len(Y) - 1):
        for i in range(len(X) - 1):
            P = [(X[i], Y[j]), (X[i + 1], Y[j]), (X[i + 1], Y[j + 1]), (X[i], Y[j + 1])]
            V = [Z[j][i], Z[j][i + 1], Z[j + 1][i + 1], Z[j + 1][i]]
            for tri in ((0, 1, 2), (0, 2, 3)):
                tp, tv = [P[t] for t in tri], [V[t] for t in tri]
                for k in range(nb):
                    a, av = _clip_linear(tp, tv, L[k], True)
                    if len(a) < 3:
                        continue
                    b, _ = _clip_linear(a, av, L[k + 1], False)
                    if len(b) < 3:
                        continue
                    ar = abs(polygon_area(b))
                    if ar > 0:
                        areas[k] += ar
                        polys[k].append(b)
    return RichResult(payload={"areas": areas, "polygons": polys, "levels": L})


def _bilinear(X, Y, Z, px, py):
    nx, ny = len(X), len(Y)
    if not (X[0] <= px <= X[-1] and Y[0] <= py <= Y[-1]):
        raise ValueError("the path leaves the grid")
    i = max(k for k in range(nx - 1) if X[k] <= px) if px < X[-1] else nx - 2
    j = max(k for k in range(ny - 1) if Y[k] <= py) if py < Y[-1] else ny - 2
    tx = (px - X[i]) / (X[i + 1] - X[i])
    ty = (py - Y[j]) / (Y[j + 1] - Y[j])
    return (
        Z[j][i] * (1 - tx) * (1 - ty)
        + Z[j][i + 1] * tx * (1 - ty)
        + Z[j + 1][i] * (1 - tx) * ty
        + Z[j + 1][i + 1] * tx * ty
    )


def elevation_profile(x, y, z, path, step: float) -> RichResult:
    r"""Elevation profile of a gridded DEM along a polyline, by bilinear interpolation every ``step``.

    Samples start at the first vertex, advance ``step`` along the path and
    include every vertex and the end point; returns the along-path
    distances, sample coordinates, elevations and the cumulative ascent and
    descent (total climb and drop).

    References
    ----------
    Burrough, P. A. and McDonnell, R. A. (1998). *Principles of Geographical
    Information Systems*. Oxford University Press, ch. 8.

    Examples
    --------
    >>> r = elevation_profile([0, 1, 2], [0, 1], [[0, 1, 2], [0, 1, 2]], [(0, 0.5), (2, 0.5)], 0.5)
    >>> r.elevation, r.ascent
    ([0.0, 0.5, 1.0, 1.5, 2.0], 2.0)
    """
    X, Y = [float(v) for v in x], [float(v) for v in y]
    Z = [[float(v) for v in row] for row in z]
    P = [(float(a), float(b)) for a, b in path]
    pts, dists = [P[0]], [0.0]
    total = 0.0
    for k in range(len(P) - 1):
        a, b = P[k], P[k + 1]
        seg = math.dist(a, b)
        t = step
        while t < seg - 1e-12:
            pts.append((a[0] + t / seg * (b[0] - a[0]), a[1] + t / seg * (b[1] - a[1])))
            dists.append(total + t)
            t += step
        total += seg
        pts.append(b)
        dists.append(total)
    el = [_bilinear(X, Y, Z, px, py) for px, py in pts]
    up = ssum(max(el[k + 1] - el[k], 0.0) for k in range(len(el) - 1))
    down = ssum(max(el[k] - el[k + 1], 0.0) for k in range(len(el) - 1))
    return RichResult(payload={"distance": dists, "points": pts, "elevation": el, "ascent": up, "descent": down})


def _seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def proximity_bands(x, y, breaks, *, points=(), lines=()) -> RichResult:
    r"""Multiple-ring proximity (buffer) bands on a grid: distance of every cell centre to the nearest feature, banded.

    Features are points and polylines; cell ``(i, j)`` at ``(x[i], y[j])``
    falls in band ``k`` when ``breaks[k-1] <= d < breaks[k]`` (band 0 below
    ``breaks[0]``; beyond the last break it is ``len(breaks)``). Areas use the
    cell spacings (regular grid assumed).

    References
    ----------
    Longley, P. A., Goodchild, M. F., Maguire, D. J. and Rhind, D. W. (2015).
    *Geographic Information Science and Systems*, 4th edn. Wiley, ch. 13 (buffering).

    Examples
    --------
    >>> r = proximity_bands([0.5, 1.5, 2.5], [0.5], [1.0, 2.0], points=[(0, 0.5)])
    >>> r.band, r.cells_per_band
    ([[0, 1, 2]], [1, 1, 1])
    """
    X, Y = [float(v) for v in x], [float(v) for v in y]
    B = [float(v) for v in breaks]
    pts = [(float(a), float(b)) for a, b in points]
    segs = []
    for ln in lines:
        L = [(float(a), float(b)) for a, b in ln]
        segs.extend((L[k], L[k + 1]) for k in range(len(L) - 1))
    dx = X[1] - X[0] if len(X) > 1 else 1.0
    dy = Y[1] - Y[0] if len(Y) > 1 else 1.0
    band, dist = [], []
    cnt = [0] * (len(B) + 1)
    for yy in Y:
        brow, drow = [], []
        for xx in X:
            d = min([math.dist((xx, yy), p) for p in pts] + [_seg_dist((xx, yy), a, b) for a, b in segs])
            k = sum(1 for v in B if d >= v)
            brow.append(k)
            drow.append(d)
            cnt[k] += 1
        band.append(brow)
        dist.append(drow)
    return RichResult(
        payload={"band": band, "distance": dist, "cells_per_band": cnt, "area_per_band": [c * dx * dy for c in cnt]}
    )


def cheatsheet() -> str:
    return (
        "polygon_boolean / polygon_area / filled_contour_bands / elevation_profile / proximity_bands -> geoprocessing."
    )
