# morie.fn -- slice s03 (rootcoder007/morie)
"""Thiessen (Voronoi) polygons.

Source consulted: Thiessen, A. H. (1911).  Precipitation averages for
large areas.  *Monthly Weather Review* 39(7), 1082-1089, which
introduces the construction: each station is assigned the region of all
points closer to it than to any other station, and the areal average is
the area-weighted mean of the station values.  The 1911 *Monthly Weather
Review* is in the public domain but was not retrievable here; the
construction is quoted in its standard published form.  It is the
Voronoi diagram of Dirichlet (1850) and Voronoi (1908) under another
name, and the module says so.

The cells are computed exactly, as intersections of half-planes with a
bounding box: for each pair (i, j) the perpendicular bisector is a
half-plane constraint, and the cell is the resulting convex polygon.
Areas follow from the shoelace formula, so the weights are exact rather
than estimated on a grid.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["thiessen_polygons"]


def _clip(poly, a, b, c):
    """Keep the part of ``poly`` where a x + b y <= c (Sutherland-Hodgman)."""
    out = []
    n = len(poly)
    for i in range(n):
        p = poly[i]
        q = poly[(i + 1) % n]
        dp = a * p[0] + b * p[1] - c
        dq = a * q[0] + b * q[1] - c
        if dp <= 0.0:
            out.append(p)
        if (dp < 0.0 < dq) or (dq < 0.0 < dp):
            t = dp / (dp - dq)
            out.append([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])])
    return out


def _area(poly):
    s = 0.0
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        s += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(s) / 2.0


def thiessen_polygons(coords, bbox=None, values=None):
    """Voronoi cells, their areas, and the area-weighted mean.

    Returns
    -------
    estimate : total area covered
    areas    : cell area per station
    weights  : areas normalised to sum to one
    areal_mean : the Thiessen average of ``values``
    """
    P = k.mat(coords)
    n = len(P)
    if bbox is None:
        xs = [P[i][0] for i in range(n)]
        ys = [P[i][1] for i in range(n)]
        mx = (max(xs) - min(xs)) or 1.0
        my = (max(ys) - min(ys)) or 1.0
        bbox = [min(xs) - 0.5 * mx, min(ys) - 0.5 * my,
                max(xs) + 0.5 * mx, max(ys) + 0.5 * my]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    areas = []
    cells = []
    for i in range(n):
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        for j in range(n):
            if i == j:
                continue
            a = 2.0 * (P[j][0] - P[i][0])
            b = 2.0 * (P[j][1] - P[i][1])
            c = (P[j][0] ** 2 + P[j][1] ** 2) - (P[i][0] ** 2 + P[i][1] ** 2)
            poly = _clip(poly, a, b, c)
            if not poly:
                break
        cells.append(poly)
        areas.append(_area(poly) if len(poly) >= 3 else 0.0)
    tot = 0.0
    for v in areas:
        tot += v
    w = [v / tot if tot > 0.0 else 0.0 for v in areas]
    am = float("nan")
    if values is not None:
        z = k.vec(values)
        s = 0.0
        for i in range(n):
            s += w[i] * z[i]
        am = s
    return RichResult(
        title="Thiessen polygons",
        summary_lines=[("stations", n), ("total area", tot)],
        payload={
            "estimate": tot,
            "areas": areas,
            "weights": w,
            "cells": cells,
            "areal_mean": am,
            "bbox": [x0, y0, x1, y1],
            "method": "Thiessen (1911) polygons by half-plane clipping; areas by the shoelace formula",
        },
    )


def cheatsheet():
    return "thnsst: Thiessen / Voronoi polygon partition"
