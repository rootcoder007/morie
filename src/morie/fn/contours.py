# morie.fn -- function file (rootcoder007/morie)
"""Contour operations on a gridded field: marching-squares isolines, filled contour bands
(isobands), clipping to a polygon, the integrated quantity between levels, contour label
placement and B-spline smoothing of contour lines."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "isolines",
    "contour_fill",
    "contour_bands",
    "contour_clip",
    "contour_quantity",
    "contour_labels",
    "contour_smooth",
]


def _interp(xa, ya, va, xb, yb, vb, level):
    t = 0.5 if vb == va else (level - va) / (vb - va)
    return (xa + t * (xb - xa), ya + t * (yb - ya))


def _cell_segments(z, xs, ys, i, j, level):
    # marching squares on cell (i, j) - (i+1, j+1); asymptotic decider for saddles
    c = [z[i][j], z[i][j + 1], z[i + 1][j + 1], z[i + 1][j]]
    p = [(xs[j], ys[i]), (xs[j + 1], ys[i]), (xs[j + 1], ys[i + 1]), (xs[j], ys[i + 1])]
    idx = sum(1 << k for k in range(4) if c[k] >= level)
    if idx in (0, 15):
        return []
    edges = []
    for k in range(4):
        a, b = k, (k + 1) % 4
        if (c[a] >= level) != (c[b] >= level):
            edges.append((k, _interp(p[a][0], p[a][1], c[a], p[b][0], p[b][1], c[b], level)))
    if len(edges) == 2:
        return [(edges[0][1], edges[1][1])]
    centre = ssum(c) / 4
    # edges 0-1-2-3 in order; pair (0,1),(2,3) or (0,3),(1,2) by the centre value
    e = {k: pt for k, pt in edges}
    if (centre >= level) == (c[0] >= level):
        return [(e[0], e[3]), (e[1], e[2])]
    return [(e[0], e[1]), (e[2], e[3])]


def _join(segments, tol=1e-12):
    # chain segments sharing endpoints into polylines
    segs = [list(s) for s in segments]
    lines = []
    while segs:
        a, b = segs.pop()
        line = [a, b]
        grown = True
        while grown:
            grown = False
            for k, (c, d) in enumerate(segs):
                if abs(c[0] - line[-1][0]) <= tol and abs(c[1] - line[-1][1]) <= tol:
                    line.append(d)
                elif abs(d[0] - line[-1][0]) <= tol and abs(d[1] - line[-1][1]) <= tol:
                    line.append(c)
                elif abs(d[0] - line[0][0]) <= tol and abs(d[1] - line[0][1]) <= tol:
                    line.insert(0, c)
                elif abs(c[0] - line[0][0]) <= tol and abs(c[1] - line[0][1]) <= tol:
                    line.insert(0, d)
                else:
                    continue
                segs.pop(k)
                grown = True
                break
        lines.append(line)
    return lines


def isolines(z, xs, ys, levels) -> RichResult:
    r"""Isolines of a gridded field by marching squares (Lorensen and Cline 1987; Maple 2003).

    ``z[i][j]`` is the value at ``(xs[j], ys[i])``; crossings are linearly
    interpolated along cell edges and saddle cells are resolved by the
    centre value. Returns one list of polylines per level.

    References
    ----------
    Lorensen, W. E. and Cline, H. E. (1987). Marching cubes. *Computer
    Graphics*, 21(4), 163-169.
    Maple, C. (2003). Geometric design and space planning using the marching
    squares and marching cube algorithms. *Proc. Geometric Modeling and
    Graphics*, 90-95.

    Examples
    --------
    >>> r = isolines([[0, 0, 0], [0, 2, 0], [0, 0, 0]], [0, 1, 2], [0, 1, 2], [1.0])
    >>> len(r.lines[0]), len(r.lines[0][0])
    (1, 5)
    """
    zz = [[float(v) for v in row] for row in z]
    out = []
    for lev in levels:
        segs = []
        for i in range(len(ys) - 1):
            for j in range(len(xs) - 1):
                segs.extend(_cell_segments(zz, xs, ys, i, j, float(lev)))
        out.append(_join(segs))
    return RichResult(payload={"lines": out, "levels": [float(v) for v in levels]})


def _clip_band(poly, z_at, lo, hi):
    # Sutherland-Hodgman clip of a polygon (with values at vertices) to lo <= value < hi
    def clip(pts, keep, val_at_edge):
        out = []
        n = len(pts)
        for k in range(n):
            (xa, ya, va), (xb, yb, vb) = pts[k], pts[(k + 1) % n]
            ina, inb = keep(va), keep(vb)
            if ina:
                out.append((xa, ya, va))
            if ina != inb:
                t = (val_at_edge - va) / (vb - va)
                out.append((xa + t * (xb - xa), ya + t * (yb - ya), val_at_edge))
        return out

    pts = clip([(x, y, v) for (x, y), v in zip(poly, z_at)], lambda v: v >= lo, lo)
    if not pts:
        return []
    pts = clip(pts, lambda v: v < hi, hi)
    return [(x, y) for x, y, _ in pts]


def contour_fill(z, xs, ys, levels) -> RichResult:
    r"""Filled contour polygons (isobands) between consecutive ``levels``.

    Each grid cell is split into its two triangles (the field is linear on a
    triangle, so the band edges are straight) and every triangle is clipped
    to ``levels[k] <= z < levels[k+1]``; a band is the list of resulting
    polygons. Areas are exact for the piecewise-linear interpolant.

    Examples
    --------
    >>> r = contour_fill([[0, 1], [1, 2]], [0, 1], [0, 1], [0, 1, 2.5])
    >>> round(r.areas[0], 12), round(r.areas[1], 12)
    (0.5, 0.5)
    """
    zz = [[float(v) for v in row] for row in z]
    bands, areas = [], []
    for k in range(len(levels) - 1):
        lo, hi = float(levels[k]), float(levels[k + 1])
        polys = []
        for i in range(len(ys) - 1):
            for j in range(len(xs) - 1):
                p = [(xs[j], ys[i]), (xs[j + 1], ys[i]), (xs[j + 1], ys[i + 1]), (xs[j], ys[i + 1])]
                v = [zz[i][j], zz[i][j + 1], zz[i + 1][j + 1], zz[i + 1][j]]
                for tri in ((0, 1, 2), (0, 2, 3)):
                    poly = _clip_band([p[a] for a in tri], [v[a] for a in tri], lo, hi)
                    if len(poly) >= 3:
                        polys.append(poly)
        bands.append(polys)
        areas.append(ssum(_area(poly) for poly in polys))
    return RichResult(payload={"bands": bands, "areas": areas, "levels": [float(v) for v in levels]})


def _area(poly):
    n = len(poly)
    return abs(ssum(poly[k][0] * poly[(k + 1) % n][1] - poly[(k + 1) % n][0] * poly[k][1] for k in range(n))) / 2


def contour_bands(z, xs, ys, levels, *, shades=None) -> RichResult:
    r"""Band shading: the isobands of :func:`contour_fill` with a shade per band.

    ``shades`` defaults to equally spaced grey levels from 0.9 (lowest band)
    to 0.2 (highest). Returns bands, areas and the shade of each band.

    Examples
    --------
    >>> r = contour_bands([[0, 1], [1, 2]], [0, 1], [0, 1], [0, 1, 2.5])
    >>> [round(s, 12) for s in r.shades]
    [0.9, 0.2]
    """
    f = contour_fill(z, xs, ys, levels)
    nb = len(levels) - 1
    if shades is None:
        shades = [0.9 - 0.7 * k / max(nb - 1, 1) for k in range(nb)]
    return RichResult(
        payload={"bands": f.bands, "areas": f.areas, "shades": [float(s) for s in shades], "levels": f.levels}
    )


def _clip_polygon(subject, clip):
    # Sutherland-Hodgman: clip a polygon to a convex polygon (counter-clockwise)
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-15

    def intersect(p, q, a, b):
        d1 = (p[0] - q[0], p[1] - q[1])
        d2 = (a[0] - b[0], a[1] - b[1])
        den = d1[0] * d2[1] - d1[1] * d2[0]
        c1 = p[0] * q[1] - p[1] * q[0]
        c2 = a[0] * b[1] - a[1] * b[0]
        return ((c1 * d2[0] - d1[0] * c2) / den, (c1 * d2[1] - d1[1] * c2) / den)

    out = list(subject)
    n = len(clip)
    for k in range(n):
        a, b = clip[k], clip[(k + 1) % n]
        inp, out = out, []
        if not inp:
            break
        for m in range(len(inp)):
            cur, prev = inp[m], inp[m - 1]
            if inside(cur, a, b):
                if not inside(prev, a, b):
                    out.append(intersect(prev, cur, a, b))
                out.append(cur)
            elif inside(prev, a, b):
                out.append(intersect(prev, cur, a, b))
    return out


def _clip_line(line, clip):
    # keep the pieces of a polyline inside a convex polygon
    def inside(p):
        n = len(clip)
        return all(
            (clip[(k + 1) % n][0] - clip[k][0]) * (p[1] - clip[k][1])
            - (clip[(k + 1) % n][1] - clip[k][1]) * (p[0] - clip[k][0])
            >= -1e-12
            for k in range(n)
        )

    pieces, cur = [], []
    for k in range(len(line) - 1):
        a, b = line[k], line[k + 1]
        # sample the segment at its intersections with the clip edges
        ts = [0.0, 1.0]
        n = len(clip)
        for m in range(n):
            c, d = clip[m], clip[(m + 1) % n]
            den = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
            if abs(den) > 1e-15:
                t = ((c[0] - a[0]) * (d[1] - c[1]) - (c[1] - a[1]) * (d[0] - c[0])) / den
                if 0 < t < 1:
                    ts.append(t)
        ts.sort()
        for u, v in zip(ts[:-1], ts[1:]):
            pa = (a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]))
            pb = (a[0] + v * (b[0] - a[0]), a[1] + v * (b[1] - a[1]))
            mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
            if inside(mid):
                if not cur:
                    cur = [pa]
                cur.append(pb)
            elif cur:
                pieces.append(cur)
                cur = []
    if cur:
        pieces.append(cur)
    return pieces


def contour_clip(lines, polygon) -> RichResult:
    r"""Clip contour polylines to a convex domain polygon (Sutherland and Hodgman 1974).

    Each polyline is cut at the polygon edges and only the pieces inside are
    kept. ``polygon`` is counter-clockwise.

    References
    ----------
    Sutherland, I. E. and Hodgman, G. W. (1974). Reentrant polygon clipping.
    *Communications of the ACM*, 17(1), 32-42.

    Examples
    --------
    >>> r = contour_clip([[(-1, 0.5), (2, 0.5)]], [(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> r.lines
    [[(0.0, 0.5), (1.0, 0.5)]]
    """
    poly = [(float(x), float(y)) for x, y in polygon]
    out = []
    for line in lines:
        out.extend(_clip_line([(float(x), float(y)) for x, y in line], poly))
    return RichResult(payload={"lines": out, "n_pieces": len(out)})


def contour_quantity(z, xs, ys, lower, upper) -> RichResult:
    r"""Area and integrated quantity of the field where ``lower <= z < upper``.

    Uses the piecewise-linear (triangle) interpolant: each triangle is
    clipped to the band and the clipped polygon is fanned into triangles;
    the integral of a linear field over a triangle is its area times the
    mean of the three vertex values (exact).

    Examples
    --------
    >>> r = contour_quantity([[0, 1], [1, 2]], [0, 1], [0, 1], 0, 3)
    >>> round(r.area, 12), round(r.integral, 12)
    (1.0, 1.0)
    """
    zz = [[float(v) for v in row] for row in z]
    area = integral = 0.0
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            p = [(xs[j], ys[i]), (xs[j + 1], ys[i]), (xs[j + 1], ys[i + 1]), (xs[j], ys[i + 1])]
            v = [zz[i][j], zz[i][j + 1], zz[i + 1][j + 1], zz[i + 1][j]]
            for tri in ((0, 1, 2), (0, 2, 3)):
                pts = [(p[a][0], p[a][1], v[a]) for a in tri]
                poly = _clip_band_values(pts, float(lower), float(upper))
                for k in range(1, len(poly) - 1):
                    fan = [poly[0], poly[k], poly[k + 1]]
                    a = _area([(x, y) for x, y, _ in fan])
                    area += a
                    integral += a * ssum(w for _, _, w in fan) / 3
    return RichResult(
        payload={"area": area, "integral": integral, "mean": integral / area if area > 0 else float("nan")}
    )


def _clip_band_values(pts, lo, hi):
    def clip(points, keep, edge):
        out = []
        n = len(points)
        for k in range(n):
            (xa, ya, va), (xb, yb, vb) = points[k], points[(k + 1) % n]
            ina, inb = keep(va), keep(vb)
            if ina:
                out.append((xa, ya, va))
            if ina != inb:
                t = (edge - va) / (vb - va)
                out.append((xa + t * (xb - xa), ya + t * (yb - ya), edge))
        return out

    out = clip(pts, lambda v: v >= lo, lo)
    return clip(out, lambda v: v < hi, hi) if out else []


def contour_labels(lines, *, min_length: float = 0.0) -> RichResult:
    r"""Label placement on contour polylines: the midpoint of the straightest run of each line.

    For every polyline the position is the midpoint of the segment whose
    neighbouring turning angles are smallest (ties to the longest segment),
    with the text angle along that segment (Dougenik 1980 style straight-run
    placement). Lines shorter than ``min_length`` get no label.

    References
    ----------
    Dougenik, J. A. (1980). WHIRLPOOL: a geometric processor for polygon
    coverage data. *Proc. Auto-Carto IV*, 304-311.

    Examples
    --------
    >>> r = contour_labels([[(0, 0), (1, 0), (2, 0), (3, 1)]])
    >>> r.positions, round(r.angles[0], 12)
    ([(0.5, 0.0)], 0.0)
    """
    pos, ang = [], []
    for line in lines:
        pts = [(float(x), float(y)) for x, y in line]
        length = ssum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1))
        if length < min_length or len(pts) < 2:
            continue
        best, key = None, None
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            seg = math.atan2(b[1] - a[1], b[0] - a[0])
            turn = 0.0
            if k > 0:
                prev = math.atan2(a[1] - pts[k - 1][1], a[0] - pts[k - 1][0])
                turn += abs(math.atan2(math.sin(seg - prev), math.cos(seg - prev)))
            if k + 2 < len(pts):
                nxt = math.atan2(pts[k + 2][1] - b[1], pts[k + 2][0] - b[0])
                turn += abs(math.atan2(math.sin(nxt - seg), math.cos(nxt - seg)))
            cand = (turn, -math.hypot(b[0] - a[0], b[1] - a[1]))
            if key is None or cand < key:
                key, best = cand, (k, seg)
        k, seg = best
        pos.append(((pts[k][0] + pts[k + 1][0]) / 2, (pts[k][1] + pts[k + 1][1]) / 2))
        ang.append(seg)
    return RichResult(payload={"positions": pos, "angles": ang})


def contour_smooth(line, *, degree: int = 3, samples: int = 50, closed: bool = False) -> RichResult:
    r"""Uniform B-spline smoothing of a contour polyline (de Boor 1978).

    The vertices are the control points of a uniform B-spline of the given
    ``degree`` evaluated at ``samples`` parameter values by de Boor's
    recurrence; open curves use a clamped knot vector so the ends are
    interpolated, closed curves wrap the control points.

    References
    ----------
    de Boor, C. (1978). *A Practical Guide to Splines*. Springer.

    Examples
    --------
    >>> r = contour_smooth([(0, 0), (1, 1), (2, 0)], degree=2, samples=3)
    >>> [(round(x, 12), round(y, 12)) for x, y in r.points]
    [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)]
    """
    P = [(float(x), float(y)) for x, y in line]
    if closed:
        P = P + P[:degree]
    n = len(P)
    if closed:
        knots = [k for k in range(n + degree + 1)]
        lo, hi = knots[degree], knots[n]
    else:
        knots = [0.0] * (degree + 1) + [float(k) for k in range(1, n - degree)] + [float(n - degree)] * (degree + 1)
        lo, hi = 0.0, float(n - degree)

    def deboor(u):
        # find knot span
        k = degree
        while k < n - 1 and not (knots[k] <= u < knots[k + 1]):
            k += 1
        if u >= hi:
            k = n - 1
        d = [P[j + k - degree] for j in range(degree + 1)]
        for r in range(1, degree + 1):
            for j in range(degree, r - 1, -1):
                i = j + k - degree
                den = knots[i + degree - r + 1] - knots[i]
                a = 0.0 if den == 0 else (u - knots[i]) / den
                d[j] = ((1 - a) * d[j - 1][0] + a * d[j][0], (1 - a) * d[j - 1][1] + a * d[j][1])
        return d[degree]

    us = [lo + (hi - lo) * s / (samples - 1) for s in range(samples)] if samples > 1 else [lo]
    pts = [deboor(min(u, hi)) for u in us]
    return RichResult(payload={"points": pts, "parameters": us})


def cheatsheet() -> str:
    return (
        "isolines / contour_fill / contour_bands / contour_clip / contour_quantity / contour_labels / "
        "contour_smooth -> contour operations on gridded fields."
    )
