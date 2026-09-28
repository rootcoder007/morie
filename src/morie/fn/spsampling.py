# morie.fn -- function file (rootcoder007/morie)
"""Spatial sampling designs and weights: uniform random and hexagonal-grid samples in a polygon, adaptive
cluster sampling with its unbiased estimators, quadtree adaptive-resolution grids, Voronoi and cell
declustering weights, spatial thinning, and design-based map quality indices."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = [
    "random_spatial_sample",
    "hexagonal_grid_sample",
    "adaptive_cluster_sample",
    "quadtree_grid",
    "voronoi_declustering_weights",
    "cell_declustering_weights",
    "spatial_thinning",
    "map_quality_indices",
]


def _inside(x, y, poly):
    """Ray casting (even-odd rule); boundary points count as inside."""
    n = len(poly)
    c = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                c = not c
            elif x == xc:
                return True
    return c


def random_spatial_sample(n: int, polygon, *, seed: int = 1) -> list:
    r"""Simple random sample of ``n`` points uniform in a polygon (rejection from its bounding box, Philox pairs).

    References
    ----------
    Brus, D. J. (2022). *Spatial Sampling with R*. CRC Press, ch. 3.

    Examples
    --------
    >>> pts = random_spatial_sample(5, [(0, 0), (2, 0), (2, 1), (0, 1)])
    >>> len(pts), all(0 <= x <= 2 and 0 <= y <= 1 for x, y in pts)
    (5, True)
    """
    poly = [(float(a), float(b)) for a, b in polygon]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    out, block = [], 0
    while len(out) < n:
        u = [float(v) for v in random_uniform(2 * max(64, 4 * n), seed=seed, stream=block)]
        block += 1
        for k in range(0, len(u), 2):
            x, y = x0 + (x1 - x0) * u[k], y0 + (y1 - y0) * u[k + 1]
            if _inside(x, y, poly):
                out.append((x, y))
                if len(out) == n:
                    break
    return out


def _rseq(a, b, by):
    k = int(math.floor((b - a) / by + 1e-10))
    return [a + i * by for i in range(k + 1)]


def hexagonal_grid_sample(polygon, cellsize: float, *, offset=(0.5, 0.5)) -> list:
    r"""Points of a hexagonal (triangular-lattice) grid inside a polygon, as ``sp::spsample(type = "hexagonal")``.

    Rows ``dy = sqrt(3)/2 cellsize`` apart, alternate rows shifted by half a
    cell, the lattice centred on the bounding box and moved by
    ``offset * (cellsize, dy)`` (``sp``'s ``genHexGrid``/``hexGrid``); points
    outside the polygon are dropped. Every point is equidistant from its six
    neighbours, giving the best coverage of the plane by equal circles.

    References
    ----------
    Pebesma, E. J. and Bivand, R. S. (2005). Classes and methods for spatial
    data in R. *R News*, 5(2), 9-13.

    Examples
    --------
    >>> len(hexagonal_grid_sample([(0, 0), (10, 0), (10, 10), (0, 10)], 2.0))
    23
    """
    poly = [(float(a), float(b)) for a, b in polygon]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    ll, ur = (min(xs), min(ys)), (max(xs), max(ys))
    dx = float(cellsize)
    dy = math.sqrt(3) * dx / 2
    x = _rseq(ll[0], ur[0] - dx / 2, dx)
    y = _rseq(ll[1], ur[1], dy)
    Y = [v for v in y for _ in x]
    base = x + [v + dx / 2 for v in x]
    X = [base[i % len(base)] for i in range(len(Y))]
    mx, my = max(X), max(Y)
    X = [v + (ur[0] - mx) / 2 + offset[0] * dx for v in X]
    Y = [v + (ur[1] - my) / 2 + offset[1] * dx * math.sqrt(3) / 2 for v in Y]
    return [(a, b) for a, b in zip(X, Y) if _inside(a, b, poly)]


def adaptive_cluster_sample(y, initial=None, *, threshold: float, n: int | None = None, seed: int = 1) -> RichResult:
    r"""Adaptive cluster sampling (Thompson 1990) on a grid population with rook neighbourhoods.

    Starting from an initial simple random sample without replacement
    (``initial`` cell indices ``(row, col)``, or ``n`` Philox draws), every
    sampled cell with ``y >= threshold`` adds its four neighbours, repeatedly.
    Networks are the connected sets of cells meeting the condition (a cell
    that does not is its own network); edge units are sampled but not part
    of a network. Returned estimators of the population mean:
    modified Hansen-Hurwitz ``mu_HH = (1/n) sum_{i in s0} w_i`` (``w_i`` the
    mean of ``i``'s network) with ``var = (N - n) / (N n (n - 1)) sum (w_i - mu_HH)^2``,
    and modified Horvitz-Thompson ``mu_HT = (1/N) sum_k y*_k / alpha_k`` over the
    distinct networks intersecting ``s0``, ``alpha_k = 1 - C(N - x_k, n) / C(N, n)``.

    References
    ----------
    Thompson, S. K. (1990). Adaptive cluster sampling. *JASA*, 85, 1050-1059.
    Thompson, S. K. (2012). *Sampling*, 3rd edn. Wiley, ch. 24.

    Examples
    --------
    >>> Y = [[0, 0, 0, 0], [0, 5, 7, 0], [0, 0, 0, 0], [0, 0, 0, 2]]
    >>> r = adaptive_cluster_sample(Y, [(1, 1), (3, 0)], threshold=1)
    >>> r.final_size, round(r.mean_hh, 6), round(r.mean_ht, 6)
    (9, 3.0, 3.103448)
    """
    Y = [[float(v) for v in row] for row in y]
    R, C = len(Y), len(Y[0])
    N = R * C
    if initial is None:
        if n is None:
            raise ValueError("give initial cells or n")
        u = [float(v) for v in random_uniform(n, seed=seed)]
        pool = list(range(N))
        s0 = []
        for t in range(n):
            s0.append(pool.pop(min(int(u[t] * len(pool)), len(pool) - 1)))
    else:
        s0 = [int(r) * C + int(c) for r, c in initial]
    n0 = len(s0)
    if len(set(s0)) != n0 or n0 < 2:
        raise ValueError("the initial sample must hold at least two distinct cells")
    meets = [Y[k // C][k % C] >= threshold for k in range(N)]

    def nbrs(k):
        r, c = divmod(k, C)
        return [a * C + b for a, b in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if 0 <= a < R and 0 <= b < C]

    net = [-1] * N
    nid = 0
    for k in range(N):
        if net[k] >= 0:
            continue
        if not meets[k]:
            net[k] = nid
            nid += 1
            continue
        stack = [k]
        net[k] = nid
        while stack:
            q = stack.pop()
            for v in nbrs(q):
                if meets[v] and net[v] < 0:
                    net[v] = nid
                    stack.append(v)
        nid += 1
    members = {}
    for k in range(N):
        members.setdefault(net[k], []).append(k)
    sampled = set()
    for k in s0:
        if meets[k]:
            for q in members[net[k]]:
                sampled.add(q)
                sampled.update(nbrs(q))
        sampled.add(k)
    w = [ssum(Y[q // C][q % C] for q in members[net[k]]) / len(members[net[k]]) for k in s0]
    mhh = ssum(w) / n0
    vhh = (N - n0) / (N * n0 * (n0 - 1)) * ssum((v - mhh) ** 2 for v in w)
    tot = 0.0
    for g in sorted({net[k] for k in s0}):
        x = len(members[g])
        alpha = 1.0 - math.comb(N - x, n0) / math.comb(N, n0)
        tot += ssum(Y[q // C][q % C] for q in members[g]) / alpha
    return RichResult(
        payload={
            "initial": [divmod(k, C) for k in s0],
            "final": sorted(divmod(k, C) for k in sampled),
            "final_size": len(sampled),
            "mean_hh": mhh,
            "var_hh": vhh,
            "mean_ht": tot / N,
        }
    )


def quadtree_grid(points, bbox, *, capacity: int = 4, max_depth: int = 8) -> RichResult:
    r"""Adaptive-resolution grid: recursive quadtree splitting of cells holding more than ``capacity`` points.

    ``bbox = (xmin, ymin, xmax, ymax)``; a point on an inner split line goes
    to the upper/right child. Returns the leaf cells ``(xmin, ymin, xmax, ymax,
    depth, count)`` in depth-first (SW, SE, NW, NE) order.

    References
    ----------
    Finkel, R. A. and Bentley, J. L. (1974). Quad trees: a data structure for
    retrieval on composite keys. *Acta Informatica*, 4, 1-9.
    Samet, H. (1984). The quadtree and related hierarchical data structures.
    *ACM Computing Surveys*, 16, 187-260.

    Examples
    --------
    >>> r = quadtree_grid([(0.1, 0.1), (0.2, 0.1), (0.15, 0.2), (0.9, 0.9)], (0, 0, 1, 1), capacity=2)
    >>> len(r.cells), [c[5] for c in r.cells]
    (10, [1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
    """
    P = [(float(a), float(b)) for a, b in points]
    out = []

    def rec(x0, y0, x1, y1, idx, depth):
        if len(idx) <= capacity or depth >= max_depth:
            out.append((x0, y0, x1, y1, depth, len(idx)))
            return
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        for a0, b0, a1, b1 in ((x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)):
            sub = [
                i
                for i in idx
                if (P[i][0] >= a0 if a0 > x0 else P[i][0] >= x0)
                and (P[i][0] < a1 if a1 < x1 else P[i][0] <= x1)
                and (P[i][1] >= b0 if b0 > y0 else P[i][1] >= y0)
                and (P[i][1] < b1 if b1 < y1 else P[i][1] <= y1)
            ]
            rec(a0, b0, a1, b1, sub, depth + 1)

    rec(*[float(v) for v in bbox], list(range(len(P))), 0)
    return RichResult(payload={"cells": out})


def _clip(poly, a, b, c):
    """Sutherland-Hodgman clip of a convex polygon to the half-plane a x + b y <= c."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        fp, fq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if fp <= 0:
            out.append(p)
        if (fp < 0 < fq) or (fq < 0 < fp):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _area(poly):
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def voronoi_declustering_weights(points, bbox) -> RichResult:
    r"""Polygonal (Voronoi) declustering weights: each point's Voronoi cell area within the box, over the total.

    Each cell is the box clipped by the perpendicular-bisector half-planes
    of all other points; isolated points get large weights and clustered
    points small ones, correcting preferential sampling in global
    statistics (the declustered mean is ``sum w_i z_i``).

    References
    ----------
    Isaaks, E. H. and Srivastava, R. M. (1989). *An Introduction to Applied
    Geostatistics*. Oxford University Press, ch. 10.

    Examples
    --------
    >>> r = voronoi_declustering_weights([(0.25, 0.5), (0.75, 0.5)], (0, 0, 1, 1))
    >>> r.weights
    [0.5, 0.5]
    """
    P = [(float(a), float(b)) for a, b in points]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    areas = []
    for i, (px, py) in enumerate(P):
        cell = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for j, (qx, qy) in enumerate(P):
            if j == i or (qx == px and qy == py):
                continue
            a, b = qx - px, qy - py
            c = 0.5 * (qx * qx + qy * qy - px * px - py * py)
            cell = _clip(cell, a, b, c)
            if not cell:
                break
        areas.append(_area(cell) if len(cell) >= 3 else 0.0)
    tot = ssum(areas)
    return RichResult(payload={"weights": [v / tot for v in areas], "areas": areas})


def cell_declustering_weights(points, cell_size: float, *, origin=(0.0, 0.0)) -> RichResult:
    r"""Cell declustering weights (Journel 1983; Deutsch 1989): ``w_i = 1 / (L n_i)`` normalised.

    ``n_i`` is the number of data in the grid cell of point ``i`` and ``L``
    the number of occupied cells, so every occupied cell carries equal total
    weight ``1/L``.

    References
    ----------
    Deutsch, C. V. (1989). DECLUS: a Fortran 77 program for determining
    optimum spatial declustering weights. *Computers and Geosciences*, 15, 325-332.

    Examples
    --------
    >>> cell_declustering_weights([(0.1, 0.1), (0.2, 0.2), (1.5, 0.5)], 1.0).weights
    [0.25, 0.25, 0.5]
    """
    P = [(float(a), float(b)) for a, b in points]
    keys = [(math.floor((x - origin[0]) / cell_size), math.floor((y - origin[1]) / cell_size)) for x, y in P]
    cnt = {}
    for k in keys:
        cnt[k] = cnt.get(k, 0) + 1
    L = len(cnt)
    return RichResult(payload={"weights": [1.0 / (L * cnt[k]) for k in keys], "occupied_cells": L})


def spatial_thinning(points, min_dist: float, *, reps: int = 10, seed: int = 1) -> RichResult:
    r"""Randomised spatial thinning of occurrence records (Aiello-Lammens et al. 2015, ``spThin``).

    In each of ``reps`` passes the records are visited in a Philox random
    order and a record is kept if it is at least ``min_dist`` from every
    record kept so far; the pass keeping the most records is returned
    (first such pass on ties). Thinning undersamples densely sampled areas
    and so reduces spatial sampling bias.

    References
    ----------
    Aiello-Lammens, M. E., Boria, R. A., Radosavljevic, A., Vilela, B. and
    Anderson, R. P. (2015). spThin: an R package for spatial thinning of
    species occurrence records. *Ecography*, 38, 541-545.

    Examples
    --------
    >>> r = spatial_thinning([(0, 0), (0.1, 0), (5, 5), (5.05, 5), (9, 0)], 1.0)
    >>> len(r.kept)
    3
    """
    P = [(float(a), float(b)) for a, b in points]
    n = len(P)
    best = None
    for rep in range(reps):
        u = [float(v) for v in random_uniform(n, seed=seed, stream=rep)]
        order = sorted(range(n), key=lambda i: (u[i], i))
        kept = []
        for i in order:
            if all(math.dist(P[i], P[j]) >= min_dist for j in kept):
                kept.append(i)
        if best is None or len(kept) > len(best):
            best = kept
    return RichResult(payload={"kept": sorted(k + 1 for k in best)})


def map_quality_indices(z, zhat, *, pi=None, N: float | None = None) -> RichResult:
    r"""Map quality indices from validation data (Brus 2022, ch. 25): ME, MAE, MSE, RMSE, r^2 and MEC.

    With prediction errors ``e_k = zhat_k - z_k``: the mean error, mean
    absolute and squared errors, their root, the squared Pearson correlation
    of predictions and observations, and the model efficiency coefficient
    ``MEC = 1 - sum e^2 / sum (z - zbar)^2`` (Janssen and Heuberger 1995).
    With inclusion probabilities ``pi`` and population size ``N`` the ME,
    MAE and MSE are pi-estimators ``(1/N) sum e_k / pi_k`` (eq. 25.8).

    References
    ----------
    Brus, D. J. (2022). *Spatial Sampling with R*. CRC Press, section 25.1.
    Janssen, P. H. M. and Heuberger, P. S. C. (1995). Calibration of
    process-oriented models. *Ecological Modelling*, 83, 55-66.

    Examples
    --------
    >>> r = map_quality_indices([1.0, 2.0, 3.0, 4.0], [1.5, 1.5, 3.5, 3.5])
    >>> r.ME, r.MSE, r.MEC
    (0.0, 0.25, 0.8)
    """
    zv, hv = [float(v) for v in z], [float(v) for v in zhat]
    n = len(zv)
    e = [hv[i] - zv[i] for i in range(n)]
    if pi is None:
        me, mae, mse = ssum(e) / n, ssum(abs(v) for v in e) / n, ssum(v * v for v in e) / n
    else:
        if N is None:
            raise ValueError("N is required with pi")
        p = [float(v) for v in pi]
        me = ssum(e[i] / p[i] for i in range(n)) / N
        mae = ssum(abs(e[i]) / p[i] for i in range(n)) / N
        mse = ssum(e[i] * e[i] / p[i] for i in range(n)) / N
    zb, hb = ssum(zv) / n, ssum(hv) / n
    sxy = ssum((zv[i] - zb) * (hv[i] - hb) for i in range(n))
    sxx = ssum((v - zb) ** 2 for v in zv)
    syy = ssum((v - hb) ** 2 for v in hv)
    return RichResult(
        payload={
            "ME": me,
            "MAE": mae,
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "r2": sxy * sxy / (sxx * syy),
            "MEC": 1.0 - ssum(v * v for v in e) / sxx,
        }
    )


def cheatsheet() -> str:
    return (
        "random_spatial_sample / hexagonal_grid_sample / adaptive_cluster_sample / quadtree_grid / "
        "voronoi_declustering_weights / cell_declustering_weights / spatial_thinning / map_quality_indices -> "
        "spatial sampling designs and weights."
    )
