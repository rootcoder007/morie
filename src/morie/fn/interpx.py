# morie.fn -- function file (rootcoder007/morie)
"""Spatial interpolation extras: tensor-product natural bicubic spline interpolation on a grid,
inverse distance weighting with line barriers (line-of-sight exclusion or shortest detour
distance), and Stewart's population potential."""

from __future__ import annotations

import heapq
import math

from . import _array_core as np
from ._qpcore import ssum

__all__ = ["bicubic_spline", "barrier_idw", "stewart_potential"]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _mat(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in r] for r in a.tolist()]


def _natural_spline(x, y, t):
    # natural cubic spline through (x, y) evaluated at t (second derivatives by the tridiagonal system)
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    m = [0.0] * n
    if n > 2:
        a = [h[i - 1] for i in range(1, n - 1)]
        b = [2 * (h[i - 1] + h[i]) for i in range(1, n - 1)]
        c = [h[i] for i in range(1, n - 1)]
        d = [6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]) for i in range(1, n - 1)]
        k = len(b)
        for i in range(1, k):
            w = a[i] / b[i - 1]
            b[i] -= w * c[i - 1]
            d[i] -= w * d[i - 1]
        sol = [0.0] * k
        sol[-1] = d[-1] / b[-1]
        for i in range(k - 2, -1, -1):
            sol[i] = (d[i] - c[i] * sol[i + 1]) / b[i]
        m = [0.0] + sol + [0.0]
    out = []
    for v in t:
        if v < x[0]:
            out.append(y[0] + ((y[1] - y[0]) / h[0] - h[0] * m[1] / 6) * (v - x[0]))
            continue
        if v > x[-1]:
            out.append(y[-1] + ((y[-1] - y[-2]) / h[-1] + h[-1] * m[-2] / 6) * (v - x[-1]))
            continue
        i = min(n - 2, max(j for j in range(n - 1) if x[j] <= v))
        hi = h[i]
        A = (x[i + 1] - v) / hi
        B = (v - x[i]) / hi
        out.append(A * y[i] + B * y[i + 1] + ((A**3 - A) * m[i] + (B**3 - B) * m[i + 1]) * hi * hi / 6)
    return out


def bicubic_spline(x, y, z, xout, yout):
    r"""Tensor-product natural bicubic spline interpolation of gridded values.

    ``z[j][i]`` is the value at ``(x[i], y[j])`` (increasing ``x``, ``y``). Each
    grid row is interpolated in ``x`` by a natural cubic spline, and the
    resulting column in ``y`` again by a natural cubic spline; the surface is
    ``C2``, passes through the data and is linear beyond the ends. Returns the
    values at the pairs ``(xout[k], yout[k])``.

    References
    ----------
    de Boor, C. (1962). Bicubic spline interpolation. *Journal of Mathematics
    and Physics* 41, 212-218.

    Examples
    --------
    >>> bicubic_spline([0, 1, 2], [0, 1], [[0, 1, 4], [1, 2, 5]], [1.0], [0.5])
    [1.5]
    """
    xv, yv = _vec(x), _vec(y)
    Z = _mat(z)
    xo, yo = _vec(xout), _vec(yout)
    out = []
    for a, b in zip(xo, yo):
        col = [_natural_spline(xv, row, [a])[0] for row in Z]
        out.append(_natural_spline(yv, col, [b])[0])
    return out


def _orient(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _cross(p, q, a, b):
    d1, d2 = _orient(a, b, p), _orient(a, b, q)
    d3, d4 = _orient(p, q, a), _orient(p, q, b)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def _visible(p, q, barriers):
    return not any(_cross(p, q, a, b) for a, b in barriers)


def barrier_idw(coords, values, targets, barriers, power=2.0, method="visibility"):
    r"""Inverse distance weighting with impermeable line barriers.

    ``barriers`` is a list of segments ``((x1, y1), (x2, y2))``.
    ``"visibility"``: only data points whose straight line to the target does
    not properly cross a barrier are used (the ArcGIS barrier rule);
    ``"path"``: distances are shortest paths that go around barriers, through
    the visibility graph of data, target and barrier end points (Dijkstra).
    Prediction ``sum w_i z_i / sum w_i``, ``w_i = d_i^-power``; an exact match
    returns its value; ``nan`` if no point is reachable.

    References
    ----------
    Shepard, D. (1968). A two-dimensional interpolation function for
    irregularly-spaced data. *Proceedings of the 23rd ACM National
    Conference*, 517-524.

    Examples
    --------
    >>> bar = [((1.0, -1.0), (1.0, 1.0))]
    >>> barrier_idw([[0, 0], [2, 0]], [10.0, 20.0], [[0.5, 0.0]], bar)
    [10.0]
    """
    P, z = _mat(coords), _vec(values)
    T = _mat(targets)
    B = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in barriers]
    ends = [list(e) for s in B for e in s]
    out = []
    for t in T:
        if method == "visibility":
            d = [math.dist(t, p) if _visible(t, p, B) else math.inf for p in P]
        elif method == "path":
            nodes = [t] + ends + P
            k = len(nodes)
            dist = [math.inf] * k
            dist[0] = 0.0
            heap = [(0.0, 0)]
            while heap:
                dd, u = heapq.heappop(heap)
                if dd > dist[u]:
                    continue
                for v in range(k):
                    if v != u and _visible(nodes[u], nodes[v], B):
                        nd = dd + math.dist(nodes[u], nodes[v])
                        if nd < dist[v]:
                            dist[v] = nd
                            heapq.heappush(heap, (nd, v))
            d = dist[1 + len(ends) :]
        else:
            raise ValueError("method must be 'visibility' or 'path'")
        exact = [i for i, v in enumerate(d) if v == 0.0]
        if exact:
            out.append(z[exact[0]])
            continue
        w = [v**-power if v < math.inf else 0.0 for v in d]
        sw = ssum(w)
        out.append(ssum(a * b for a, b in zip(w, z)) / sw if sw > 0 else math.nan)
    return out


def stewart_potential(coords, masses, beta=1.0, self_distance=None):
    r"""Stewart's population potential ``V_i = sum_j M_j / d_ij^beta``.

    The own mass enters with ``self_distance`` (e.g. the radius of the unit's
    area) or is excluded when it is ``None``.

    References
    ----------
    Stewart, J. Q. (1947). Empirical mathematical rules concerning the
    distribution and equilibrium of population. *Geographical Review* 37,
    461-485.

    Examples
    --------
    >>> stewart_potential([[0, 0], [3, 4]], [100.0, 50.0])
    [10.0, 20.0]
    """
    P, M = _mat(coords), _vec(masses)
    out = []
    for i, p in enumerate(P):
        tot = 0.0
        for j, q in enumerate(P):
            if i == j:
                if self_distance is not None:
                    tot += M[j] / self_distance**beta
                continue
            tot += M[j] / math.dist(p, q) ** beta
        out.append(tot)
    return out


def cheatsheet() -> str:
    return "bicubic_spline / barrier_idw / stewart_potential -> interpolation extras."
