# morie.fn -- function file (rootcoder007/morie)
"""Grid generation and grid-based sampling: square and hexagonal cell centres over a bounding box,
triangular meshes, transfinite (Coons) curvilinear grids from four boundary curves, systematic
grid samples with a random start, and space-time sampling designs (static, synchronous,
static-synchronous and rotating panels)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["regular_grid", "triangular_grid", "transfinite_grid", "systematic_sample", "space_time_sample"]


def _pts(P):
    return [[float(v) for v in r] for r in np.asarray(P, dtype=float).tolist()]


def regular_grid(bbox, cellsize, shape="square"):
    r"""Cell centres of a square or hexagonal grid covering ``bbox = (xmin, ymin, xmax, ymax)``.

    Square: ``ceil((xmax - xmin)/d)`` by ``ceil((ymax - ymin)/d)`` cells from
    the lower-left corner, centres at ``xmin + (i + 1/2) d``, row by row from the
    bottom (``sf::st_make_grid(what = "centers")``). Hexagonal (pointy-topped,
    centre spacing ``d``): rows ``d sqrt(3)/2`` apart, odd rows shifted by
    ``d/2``, keeping centres within half a cell of the box.

    References
    ----------
    Birch, C. P. D., Oom, S. P. and Beecham, J. A. (2007). Rectangular and
    hexagonal grids used for observation, experiment and simulation in
    ecology. *Ecological Modelling* 206, 347-359.

    Examples
    --------
    >>> regular_grid((0, 0, 2, 1), 1.0)
    [[0.5, 0.5], [1.5, 0.5]]
    """
    x0, y0, x1, y1 = [float(v) for v in bbox]
    d = float(cellsize)
    out = []
    if shape == "square":
        nx, ny = math.ceil((x1 - x0) / d - 1e-12), math.ceil((y1 - y0) / d - 1e-12)
        for j in range(ny):
            for i in range(nx):
                out.append([x0 + (i + 0.5) * d, y0 + (j + 0.5) * d])
        return out
    if shape == "hexagon":
        dy = d * math.sqrt(3) / 2
        j = 0
        while y0 + j * dy <= y1 + dy / 2:
            off = d / 2 if j % 2 else 0.0
            i = 0
            while x0 + off + i * d <= x1 + d / 2:
                out.append([x0 + off + i * d, y0 + j * dy])
                i += 1
            j += 1
        return out
    raise ValueError("shape must be 'square' or 'hexagon'")


def triangular_grid(bbox, side):
    r"""Equilateral triangular mesh over ``bbox``: vertices on a triangular lattice and the triangles.

    Vertex rows are ``side sqrt(3)/2`` apart with odd rows shifted by ``side/2``;
    each pair of adjacent rows is split into up- and down-pointing triangles
    (vertex index triples, counter-clockwise).

    Examples
    --------
    >>> r = triangular_grid((0, 0, 1, 0.9), 1.0)
    >>> len(r.vertices), r.triangles
    (6, [[0, 1, 2], [1, 3, 2], [2, 5, 4], [2, 3, 5]])
    """
    x0, y0, x1, y1 = [float(v) for v in bbox]
    h = side * math.sqrt(3) / 2
    ny = math.ceil((y1 - y0) / h - 1e-12) + 1
    nx = math.ceil((x1 - x0) / side - 1e-12) + 1
    V = []
    for j in range(ny):
        off = side / 2 if j % 2 else 0.0
        for i in range(nx):
            V.append([x0 + off + i * side, y0 + j * h])
    T = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b = j * nx + i, j * nx + i + 1
            c, d = (j + 1) * nx + i, (j + 1) * nx + i + 1
            if j % 2 == 0:
                T += [[a, b, c], [b, d, c]]
            else:
                T += [[a, d, c], [a, b, d]]
    return RichResult(payload={"vertices": V, "triangles": T})


def transfinite_grid(bottom, top, left, right):
    r"""Curvilinear grid by transfinite (Coons patch) interpolation of four boundary curves.

    ``bottom`` and ``top`` are sampled at ``nu`` points, ``left`` and ``right``
    at ``nv`` points, sharing the corners. With ``u_i = i/(nu - 1)``,
    ``v_j = j/(nv - 1)``: ``P(u, v) = (1 - v) B(u) + v T(u) + (1 - u) L(v) + u R(v)
    - [(1 - u)(1 - v) P00 + u (1 - v) P10 + (1 - u) v P01 + u v P11]``.
    Returns ``nv`` rows of ``nu`` points.

    References
    ----------
    Gordon, W. J. and Hall, C. A. (1973). Construction of curvilinear
    co-ordinate systems and applications to mesh generation. *International
    Journal for Numerical Methods in Engineering* 7, 461-477.

    Examples
    --------
    >>> g = transfinite_grid([[0, 0], [1, 0]], [[0, 1], [1, 1]], [[0, 0], [0, 1]], [[1, 0], [1, 1]])
    >>> g
    [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]
    """
    B, T, L, R = _pts(bottom), _pts(top), _pts(left), _pts(right)
    nu, nv = len(B), len(L)
    P00, P10, P01, P11 = B[0], B[-1], T[0], T[-1]
    out = []
    for j in range(nv):
        v = j / (nv - 1)
        row = []
        for i in range(nu):
            u = i / (nu - 1)
            row.append(
                [
                    (1 - v) * B[i][c]
                    + v * T[i][c]
                    + (1 - u) * L[j][c]
                    + u * R[j][c]
                    - ((1 - u) * (1 - v) * P00[c] + u * (1 - v) * P10[c] + (1 - u) * v * P01[c] + u * v * P11[c])
                    for c in range(2)
                ]
            )
        out.append(row)
    return out


def _inside(p, poly):
    x, y = p
    c = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def systematic_sample(bbox, spacing, seed=0, polygon=None):
    r"""Systematic (square grid) sample with a uniformly random start.

    The start ``(x0 + U1 d, y0 + U2 d)`` (Philox uniforms) and then every ``d``
    in both directions within ``bbox``; with ``polygon`` only points inside
    (ray casting) are kept. Every point of the box has inclusion density
    ``1/d^2``.

    References
    ----------
    Brus, D. J. (2022). *Spatial Sampling with R*. CRC Press, chapter 5.

    Examples
    --------
    >>> len(systematic_sample((0, 0, 10, 10), 2.0))
    25
    """
    x0, y0, x1, y1 = [float(v) for v in bbox]
    u = [float(v) for v in random_uniform(2, seed=seed)]
    sx, sy = x0 + u[0] * spacing, y0 + u[1] * spacing
    out = []
    y = sy
    while y <= y1:
        x = sx
        while x <= x1:
            out.append([x, y])
            x += spacing
        y += spacing
    if polygon is not None:
        poly = _pts(polygon)
        out = [p for p in out if _inside(p, poly)]
    return out


def _srs(N, n, u):
    return sorted(sorted(range(N), key=lambda i: (u[i], i))[:n])


def space_time_sample(n_population, n_sample, n_times, design="static", n_static=None, rotation=2, seed=0):
    r"""Space-time sampling designs over ``n_times`` occasions (unit indices per occasion).

    ``"static"``: one simple random sample revisited every time;
    ``"synchronous"``: an independent sample each time;
    ``"static_synchronous"``: a static panel of ``n_static`` units plus an
    independent synchronous remainder; ``"rotating"``: a rotating panel in
    which ``n_sample / rotation`` units are replaced each occasion and each
    unit stays ``rotation`` occasions. Random selection sorts Philox
    uniforms.

    References
    ----------
    de Gruijter, J. J., Brus, D. J., Bierkens, M. F. P. and Knotters, M.
    (2006). *Sampling for Natural Resource Monitoring*. Springer, chapter 14.

    Examples
    --------
    >>> s = space_time_sample(20, 4, 3, "static")
    >>> s[0] == s[1] == s[2], len(s[0])
    (True, 4)
    """
    N, n, T = int(n_population), int(n_sample), int(n_times)
    u = [float(v) for v in random_uniform(N * (T + 1), seed=seed)]
    block = lambda t: u[t * N : (t + 1) * N]  # noqa: E731
    if design == "static":
        s = _srs(N, n, block(0))
        return [list(s) for _ in range(T)]
    if design == "synchronous":
        return [_srs(N, n, block(t)) for t in range(T)]
    if design == "static_synchronous":
        k = n // 2 if n_static is None else int(n_static)
        st = _srs(N, k, block(0))
        rest = [i for i in range(N) if i not in st]
        out = []
        for t in range(T):
            b = block(t + 1)
            pick = sorted(rest, key=lambda i: (b[i], i))[: n - k]
            out.append(sorted(st + pick))
        return out
    if design == "rotating":
        g = n // rotation
        order = sorted(range(N), key=lambda i: (u[i], i))
        groups = [order[k * g : (k + 1) * g] for k in range(T + rotation - 1)]
        if len(order) < g * (T + rotation - 1):
            raise ValueError("population too small for this rotating panel")
        return [sorted(sum(groups[t : t + rotation], [])) for t in range(T)]
    raise ValueError("design must be static, synchronous, static_synchronous or rotating")


def cheatsheet() -> str:
    return "regular_grid / triangular_grid / transfinite_grid / systematic_sample / space_time_sample -> grids."
