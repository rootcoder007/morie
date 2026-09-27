# morie.fn -- function file (rootcoder007/morie)
"""Voronoi (Dirichlet) tessellation clipped to a convex window."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._qpcore import ssum


def _clip(poly, a, b):
    """Sutherland-Hodgman: keep the part of ``poly`` with a . u <= b."""
    out = []
    m = len(poly)
    for k in range(m):
        p, q = poly[k], poly[(k + 1) % m]
        fp = a[0] * p[0] + a[1] * p[1] - b
        fq = a[0] * q[0] + a[1] * q[1] - b
        if fp <= 0:
            out.append(p)
        if (fp < 0 < fq) or (fq < 0 < fp):
            t = fp / (fp - fq)
            out.append([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])])
    return out


def _area(poly):
    m = len(poly)
    return 0.5 * ssum(poly[k][0] * poly[(k + 1) % m][1] - poly[(k + 1) % m][0] * poly[k][1] for k in range(m))


def _centroid(poly, area):
    m = len(poly)
    cx = cy = 0.0
    for k in range(m):
        p, q = poly[k], poly[(k + 1) % m]
        c = p[0] * q[1] - q[0] * p[1]
        cx += (p[0] + q[0]) * c
        cy += (p[1] + q[1]) * c
    return [cx / (6.0 * area), cy / (6.0 * area)]


def voronoi_cells(points, window) -> DescriptiveResult:
    """Voronoi cells of a point pattern inside a convex window.

    The cell of ``x_i`` is the window intersected with the half-planes
    ``(x_j - x_i) . (u - (x_i + x_j) / 2) <= 0`` for every other point
    ``x_j``, computed by Sutherland-Hodgman clipping. Returned with the
    cell areas, the Voronoi intensity estimate ``1 / area`` at each point
    (Barr and Schoenberg 2010; ``spatstat.explore::densityVoronoi`` with
    ``f = 1`` evaluated at the points), the perimeters and centroids of the
    cells, the Voronoi edges between points (cells that share an edge of
    positive length: the Delaunay neighbours) with their lengths, the
    binary adjacency matrix, the number of neighbours of each cell and the
    Voronoi entropy ``-sum_k P_k log P_k`` of the distribution of neighbour
    counts (Bormashenko et al. 2018). Areas and neighbours equal
    ``deldir::deldir(x, y, rw = window, digits = 15)``.

    :param points: (n, 2) distinct points inside ``window``.
    :param window: Rectangle ``(xmin, xmax, ymin, ymax)`` or a convex
        polygon as a list of vertices in counter-clockwise order.
    :return: DescriptiveResult; ``value`` is the cell areas; ``extra`` has
        ``cells`` (vertex lists, counter-clockwise), ``intensity``,
        ``perimeters``, ``centroids``, ``edges`` (``[i, j, length]`` with
        ``i < j``), ``neighbours`` (index lists), ``adjacency``,
        ``neighbour_counts`` and ``entropy``.

    References
    ----------
    Okabe, A., Boots, B., Sugihara, K. and Chiu, S. N. (2000). Spatial
    Tessellations, 2nd ed. Wiley, Ch. 2.

    Barr, C. D. and Schoenberg, F. P. (2010). On the Voronoi estimator for
    the intensity of an inhomogeneous planar Poisson process. Biometrika
    97, 977-984.

    Sutherland, I. E. and Hodgman, G. W. (1974). Reentrant polygon
    clipping. Communications of the ACM 17, 32-42.

    Bormashenko, E. et al. (2018). Characterization of self-assembled 2D
    patterns with Voronoi entropy. Entropy 20, 956.

    Examples
    --------
    >>> r = voronoi_cells([[0.25, 0.5], [0.75, 0.5]], (0, 1, 0, 1))
    >>> r.value, r.extra["neighbours"]
    ([0.5, 0.5], [[1], [0]])
    """
    P = [[float(v) for v in p] for p in points]
    if len(window) == 4 and not hasattr(window[0], "__len__"):
        x0, x1, y0, y1 = (float(v) for v in window)
        W = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    else:
        W = [[float(v) for v in p] for p in window]
    n = len(P)
    cells, areas, nbrs = [], [], []
    for i in range(n):
        poly = [list(v) for v in W]
        for j in range(n):
            if j == i:
                continue
            a = [P[j][0] - P[i][0], P[j][1] - P[i][1]]
            b = a[0] * (P[i][0] + P[j][0]) / 2 + a[1] * (P[i][1] + P[j][1]) / 2
            poly = _clip(poly, a, b)
            if not poly:
                break
        cells.append(poly)
        areas.append(_area(poly) if len(poly) >= 3 else 0.0)
    scale = max(abs(v) for p in W for v in p) or 1.0
    edges = []
    for i in range(n):
        near = []
        for j in range(n):
            if j == i:
                continue
            a = [P[j][0] - P[i][0], P[j][1] - P[i][1]]
            b = a[0] * (P[i][0] + P[j][0]) / 2 + a[1] * (P[i][1] + P[j][1]) / 2
            norm = (a[0] ** 2 + a[1] ** 2) ** 0.5
            on = [v for v in cells[i] if abs(a[0] * v[0] + a[1] * v[1] - b) <= 1e-9 * scale * norm]
            length = (
                max(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 for p in on for q in on) if len(on) >= 2 else 0.0
            )
            if length > 1e-9 * scale:
                near.append(j)
                if i < j:
                    edges.append([i, j, length])
        nbrs.append(near)
    perim = [
        ssum(
            ((c[k][0] - c[(k + 1) % len(c)][0]) ** 2 + (c[k][1] - c[(k + 1) % len(c)][1]) ** 2) ** 0.5
            for k in range(len(c))
        )
        for c in cells
    ]
    counts = [len(v) for v in nbrs]
    freq = {}
    for k in counts:
        freq[k] = freq.get(k, 0) + 1
    entropy = 0.0 - ssum(f / n * math.log(f / n) for f in freq.values())
    return DescriptiveResult(
        name="voronoi_cells",
        value=areas,
        extra={
            "cells": cells,
            "intensity": [1.0 / a if a > 0 else float("inf") for a in areas],
            "perimeters": perim,
            "centroids": [_centroid(c, a) if a > 0 else None for c, a in zip(cells, areas)],
            "edges": edges,
            "neighbours": nbrs,
            "adjacency": [[1 if j in nb else 0 for j in range(n)] for nb in nbrs],
            "neighbour_counts": counts,
            "entropy": entropy,
        },
    )


vorcel = voronoi_cells


def cheatsheet() -> str:
    return "voronoi_cells(points, window) -> clipped Voronoi cells, areas, Voronoi intensity, Delaunay neighbours"
