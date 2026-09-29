# morie.fn -- function file (rootcoder007/morie)
"""Meshes: 3-D Delaunay tetrahedralisation by Bowyer-Watson, 3-D Voronoi cells from its dual (cell
vertices, faces and volumes), and Ruppert's Delaunay refinement of a planar point set to a
quality triangulation with a minimum-angle guarantee."""

from __future__ import annotations

import math

from ._qpcore import solve, ssum
from ._richresult import RichResult

__all__ = ["delaunay_3d", "voronoi_3d", "ruppert_refine"]


def _circumsphere(P):
    # centre c solves 2 (p_k - p_0) . c = |p_k|^2 - |p_0|^2
    d = len(P[0])
    A = [[2.0 * (P[k][a] - P[0][a]) for a in range(d)] for k in range(1, d + 1)]
    b = [ssum(v * v for v in P[k]) - ssum(v * v for v in P[0]) for k in range(1, d + 1)]
    try:
        c = solve(A, b)
    except ZeroDivisionError:
        return None, math.inf
    return c, ssum((c[a] - P[0][a]) ** 2 for a in range(d))


def _bowyer_watson(points, d):
    pts = [[float(v) for v in p] for p in points]
    n = len(pts)
    lo = [min(p[a] for p in pts) for a in range(d)]
    hi = [max(p[a] for p in pts) for a in range(d)]
    span = max(hi[a] - lo[a] for a in range(d)) or 1.0
    mid = [(lo[a] + hi[a]) / 2 for a in range(d)]
    big = 50.0 * span
    if d == 2:
        sup = [[mid[0] - 2 * big, mid[1] - big], [mid[0] + 2 * big, mid[1] - big], [mid[0], mid[1] + 2 * big]]
    else:
        sup = [
            [mid[0] - big, mid[1] - big, mid[2] - big],
            [mid[0] + 3 * big, mid[1] - big, mid[2] - big],
            [mid[0] - big, mid[1] + 3 * big, mid[2] - big],
            [mid[0] - big, mid[1] - big, mid[2] + 3 * big],
        ]
    allp = pts + sup
    simp = [tuple(range(n, n + d + 1))]
    cache = {simp[0]: _circumsphere([allp[i] for i in simp[0]])}
    for i in range(n):
        p = allp[i]
        bad = []
        for s in simp:
            c, r2 = cache[s]
            if c is not None and ssum((p[a] - c[a]) ** 2 for a in range(d)) < r2 * (1 - 1e-12):
                bad.append(s)
        faces = {}
        for s in bad:
            for k in range(d + 1):
                f = tuple(sorted(s[:k] + s[k + 1 :]))
                faces[f] = faces.get(f, 0) + 1
        badset = set(bad)
        simp = [s for s in simp if s not in badset]
        for f, cnt in sorted(faces.items()):
            if cnt == 1:
                s = tuple(sorted(f + (i,)))
                simp.append(s)
                cache[s] = _circumsphere([allp[q] for q in s])
    return sorted(s for s in simp if all(q < n for q in s)), pts


def delaunay_3d(points) -> RichResult:
    r"""Delaunay tetrahedralisation of 3-D points by the Bowyer-Watson algorithm.

    Points are inserted in order into a super-tetrahedron; the tetrahedra
    whose circumspheres contain the new point are removed and the cavity is
    re-triangulated from its boundary faces (Bowyer 1981; Watson 1981).
    Tetrahedra touching the super-tetrahedron are dropped. Returns sorted
    0-based index quadruples and their circumcentres. The tetrahedralisation
    is unique for points in general position (no five cospherical).

    References
    ----------
    Bowyer, A. (1981). Computing Dirichlet tessellations. Computer J. 24,
    162-166. Watson, D. F. (1981). Computing the n-dimensional Delaunay
    tessellation with application to Voronoi polytopes. Computer J. 24, 167-172.

    Examples
    --------
    >>> r = delaunay_3d([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1.2)])
    >>> r.tetrahedra
    [[0, 1, 2, 3], [1, 2, 3, 4]]
    """
    tets, pts = _bowyer_watson(points, 3)
    cc = [_circumsphere([pts[q] for q in t])[0] for t in tets]
    return RichResult(payload={"tetrahedra": [list(t) for t in tets], "circumcenters": cc})


def _polygon_area(P, axis):
    # order the points by angle around the axis direction through their centroid, then fan-triangulate
    c = [ssum(p[a] for p in P) / len(P) for a in range(3)]
    ax = axis
    tmp = [1.0, 0.0, 0.0] if abs(ax[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = [ax[1] * tmp[2] - ax[2] * tmp[1], ax[2] * tmp[0] - ax[0] * tmp[2], ax[0] * tmp[1] - ax[1] * tmp[0]]
    nu = math.sqrt(ssum(v * v for v in u))
    u = [v / nu for v in u]
    w = [ax[1] * u[2] - ax[2] * u[1], ax[2] * u[0] - ax[0] * u[2], ax[0] * u[1] - ax[1] * u[0]]
    ang = [
        math.atan2(ssum((p[a] - c[a]) * w[a] for a in range(3)), ssum((p[a] - c[a]) * u[a] for a in range(3)))
        for p in P
    ]
    order = sorted(range(len(P)), key=lambda q: ang[q])
    Q = [P[q] for q in order]
    area = 0.0
    for k in range(len(Q)):
        a = [Q[k][t] - c[t] for t in range(3)]
        b = [Q[(k + 1) % len(Q)][t] - c[t] for t in range(3)]
        cr = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        area += 0.5 * math.sqrt(ssum(v * v for v in cr))
    return area


def voronoi_3d(points) -> RichResult:
    r"""3-D Voronoi tessellation as the dual of the Delaunay tetrahedralisation.

    Cell ``i`` has as vertices the circumcentres of the tetrahedra incident
    to point ``i``; it is unbounded when ``i`` lies on the convex hull (a
    face belonging to a single tetrahedron). For bounded cells the face
    shared with neighbour ``j`` is the polygon of circumcentres of the
    tetrahedra around edge ``ij`` (ordered by angle about the edge) and the
    volume is ``sum_j (1/3) A_ij |x_i - x_j| / 2`` (pyramids on the faces).

    References
    ----------
    Okabe, A., Boots, B., Sugihara, K. and Chiu, S. N. (2000). Spatial
    Tessellations: Concepts and Applications of Voronoi Diagrams, 2nd ed.,
    ch. 2-4. Aurenhammer, F. (1991). Voronoi diagrams - a survey of a
    fundamental geometric data structure. ACM Computing Surveys 23, 345-405.

    Examples
    --------
    >>> pts = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    >>> r = voronoi_3d(pts)
    >>> round(r.volumes[0], 12), r.bounded[1]
    (1.0, False)
    """
    d = delaunay_3d(points)
    pts = [[float(v) for v in p] for p in points]
    n = len(pts)
    tets = [tuple(t) for t in d.tetrahedra]
    cc = d.circumcenters
    fc = {}
    for t in tets:
        for k in range(4):
            f = tuple(sorted(t[:k] + t[k + 1 :]))
            fc[f] = fc.get(f, 0) + 1
    hull = set(q for f, c in fc.items() if c == 1 for q in f)
    bounded = [i not in hull for i in range(n)]
    verts = [[cc[k] for k, t in enumerate(tets) if i in t] for i in range(n)]
    vol = [math.inf] * n
    for i in range(n):
        if not bounded[i]:
            continue
        nb = sorted(set(q for t in tets if i in t for q in t if q != i))
        total = 0.0
        for j in nb:
            P = [cc[k] for k, t in enumerate(tets) if i in t and j in t]
            if len(P) < 3:
                continue
            dij = math.sqrt(ssum((pts[i][a] - pts[j][a]) ** 2 for a in range(3)))
            axis = [(pts[j][a] - pts[i][a]) / dij for a in range(3)]
            total += _polygon_area(P, axis) * dij / 6.0
        vol[i] = total
    return RichResult(payload={"vertices": verts, "bounded": bounded, "volumes": vol})


def _hull2d(P):
    idx = sorted(range(len(P)), key=lambda i: (P[i][0], P[i][1]))

    def cross(o, a, b):
        return (P[a][0] - P[o][0]) * (P[b][1] - P[o][1]) - (P[a][1] - P[o][1]) * (P[b][0] - P[o][0])

    lower, upper = [], []
    for i in idx:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    for i in reversed(idx):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def ruppert_refine(points, *, min_angle: float = 20.0, max_points: int = 500) -> RichResult:
    r"""Ruppert's Delaunay refinement of the triangulation of a planar point set.

    The boundary segments are the convex hull edges. Repeatedly: a boundary
    segment whose diametral circle strictly contains a vertex is split at
    its midpoint; otherwise the triangle with the largest circumradius to
    shortest-edge ratio above ``B = 1 / (2 sin(min_angle))`` gets its
    circumcentre inserted, unless that circumcentre encroaches a segment,
    which is split instead. Stops when no triangle is skinny (all angles at
    least ``min_angle``, guaranteed for ``min_angle <= 20.7`` degrees) or at
    ``max_points``.

    References
    ----------
    Ruppert, J. (1995). A Delaunay refinement algorithm for quality
    2-dimensional mesh generation. J. Algorithms 18, 548-585. Shewchuk, J. R.
    (2002). Delaunay refinement algorithms for triangular mesh generation.
    Computational Geometry 22, 21-74.

    Examples
    --------
    >>> r = ruppert_refine([(0, 0), (4, 0), (4, 1), (0, 1)])
    >>> r.min_angle >= 20.0
    True
    """
    P = [[float(a), float(b)] for a, b in points]
    hull = _hull2d(P)
    segs = [(hull[k], hull[(k + 1) % len(hull)]) for k in range(len(hull))]
    B = 1.0 / (2.0 * math.sin(math.radians(min_angle)))

    def encroached(s, q):
        a, b = P[s[0]], P[s[1]]
        c = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
        r2 = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / 4
        return (q[0] - c[0]) ** 2 + (q[1] - c[1]) ** 2 < r2 * (1 - 1e-12)

    def split(k):
        a, b = segs[k]
        P.append([(P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2])
        m = len(P) - 1
        segs[k : k + 1] = [(a, m), (m, b)]

    while len(P) < max_points:
        tris, _ = _bowyer_watson(P, 2)
        enc = next(
            (k for k, s in enumerate(segs) if any(encroached(s, P[q]) for q in range(len(P)) if q not in s)), None
        )
        if enc is not None:
            split(enc)
            continue
        worst, wr, wc = None, B, None
        for t in tris:
            c, r2 = _circumsphere([P[q] for q in t])
            e = min(
                (P[t[a]][0] - P[t[b]][0]) ** 2 + (P[t[a]][1] - P[t[b]][1]) ** 2 for a, b in ((0, 1), (1, 2), (0, 2))
            )
            ratio = math.sqrt(r2 / e)
            if ratio > wr + 1e-12:
                worst, wr, wc = t, ratio, c
        if worst is None:
            break
        enc = next((k for k, s in enumerate(segs) if encroached(s, wc)), None)
        if enc is not None:
            split(enc)
        else:
            P.append([wc[0], wc[1]])
    tris, _ = _bowyer_watson(P, 2)

    def angles(t):
        out = []
        for k in range(3):
            a, b, c = P[t[k]], P[t[(k + 1) % 3]], P[t[(k + 2) % 3]]
            v1 = [b[0] - a[0], b[1] - a[1]]
            v2 = [c[0] - a[0], c[1] - a[1]]
            cs = (v1[0] * v2[0] + v1[1] * v2[1]) / math.sqrt((v1[0] ** 2 + v1[1] ** 2) * (v2[0] ** 2 + v2[1] ** 2))
            out.append(math.degrees(math.acos(max(-1.0, min(1.0, cs)))))
        return out

    mn = min(min(angles(t)) for t in tris)
    return RichResult(
        payload={
            "points": P,
            "triangles": [list(t) for t in tris],
            "segments": [list(s) for s in segs],
            "min_angle": mn,
        }
    )


def cheatsheet() -> str:
    return "delaunay_3d / voronoi_3d / ruppert_refine -> 3-D Delaunay and Voronoi, Ruppert Delaunay refinement."
