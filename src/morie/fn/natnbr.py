# morie.fn -- function file (rootcoder007/morie)
"""Natural-neighbour interpolation: Sibson (area-stealing) and Laplace (non-Sibsonian) coordinates, the
gradient-enhanced interpolant, leave-one-out cross-validation, the natural-neighbour variance, the bounds
of the interpolant and its convex-hull domain."""

from __future__ import annotations

import math

from ._qpcore import solve, ssum
from ._richresult import RichResult

__all__ = [
    "natural_neighbour_weights",
    "nn_interpolate",
    "nn_gradient_interpolate",
    "nn_cross_validation",
    "nn_in_domain",
]


def _pts(points):
    return [(float(a), float(b)) for a, b in points]


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
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _area(poly):
    m = len(poly)
    return 0.5 * ssum(poly[k][0] * poly[(k + 1) % m][1] - poly[(k + 1) % m][0] * poly[k][1] for k in range(m))


def _halfplane(p, q):
    """Half-plane of points at least as close to ``p`` as to ``q``: a . u <= b."""
    a = (q[0] - p[0], q[1] - p[1])
    return a, a[0] * (p[0] + q[0]) / 2 + a[1] * (p[1] + q[1]) / 2


def _window(P, x, scale=10.0):
    xs = [p[0] for p in P] + [x[0]]
    ys = [p[1] for p in P] + [x[1]]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    lo_x, hi_x = min(xs) - scale * span, max(xs) + scale * span
    lo_y, hi_y = min(ys) - scale * span, max(ys) + scale * span
    return [(lo_x, lo_y), (hi_x, lo_y), (hi_x, hi_y), (lo_x, hi_y)]


def _cell(P, x):
    """Voronoi cell of x among P plus x; the window grows until the cell no longer reaches it (bounded cell)."""
    for scale in (10.0, 1e3, 1e5, 1e7):
        W = _window(P, x, scale)
        cell = W
        for q in P:
            a, b = _halfplane(x, q)
            cell = _clip(cell, a, b)
        tol = 1e-9 * (W[1][0] - W[0][0])
        if not any(
            abs(v[0] - W[0][0]) < tol
            or abs(v[0] - W[1][0]) < tol
            or abs(v[1] - W[0][1]) < tol
            or abs(v[1] - W[2][1]) < tol
            for v in cell
        ):
            break
    return cell


def _hull(P):
    S = sorted(set(P))
    if len(S) <= 2:
        return S

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, up = [], []
    for p in S:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(S):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def _in_hull(H, x, tol=1e-12):
    n = len(H)
    for k in range(n):
        p, q = H[k], H[(k + 1) % n]
        if (q[0] - p[0]) * (x[1] - p[1]) - (q[1] - p[1]) * (x[0] - p[0]) < -tol * max(1.0, abs(x[0]) + abs(x[1])):
            return False
    return True


def _weights(P, x, method):
    for i, p in enumerate(P):
        if p == x:
            return {i: 1.0}
    cell = _cell(P, x)
    ax = abs(_area(cell))
    w = {}
    if method == "sibson":
        for i, p in enumerate(P):
            reg = cell
            for j, q in enumerate(P):
                if j != i and reg:
                    a, b = _halfplane(p, q)
                    reg = _clip(reg, a, b)
            if len(reg) >= 3:
                s = abs(_area(reg))
                if s > 1e-15 * ax:
                    w[i] = s / ax
        return w
    if method != "laplace":
        raise ValueError("method must be sibson or laplace")
    m = len(cell)
    for k in range(m):
        u, v = cell[k], cell[(k + 1) % m]
        L = math.hypot(v[0] - u[0], v[1] - u[1])
        if L <= 0:
            continue
        mid = ((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)
        dx = math.hypot(mid[0] - x[0], mid[1] - x[1])
        best = min(range(len(P)), key=lambda j: (abs(math.hypot(mid[0] - P[j][0], mid[1] - P[j][1]) - dx), j))
        if abs(math.hypot(mid[0] - P[best][0], mid[1] - P[best][1]) - dx) <= 1e-9 * max(1.0, dx):
            w[best] = w.get(best, 0.0) + L / math.hypot(P[best][0] - x[0], P[best][1] - x[1])
    tot = ssum(w.values())
    return {i: v / tot for i, v in w.items()}


def natural_neighbour_weights(points, x, *, method: str = "sibson") -> RichResult:
    r"""Natural-neighbour coordinates of ``x`` with respect to ``points``.

    ``sibson`` (Sibson 1981): the area each data point's Voronoi cell loses to
    the cell of ``x`` when ``x`` is inserted, over the area of that cell;
    ``laplace`` (non-Sibsonian; Belikov et al. 1997): the length of the
    Voronoi facet shared by ``x`` and ``x_i`` divided by ``|x - x_i|``,
    normalised. Both are positive only on the natural neighbours, sum to 1
    and reproduce linear functions exactly (``sum w_i x_i = x``) for ``x``
    inside the convex hull; at a data point the weight is 1 on it.

    References
    ----------
    Sibson, R. (1981). A brief description of natural neighbour
    interpolation. In V. Barnett (ed.), *Interpreting Multivariate Data*,
    21-36. Wiley.
    Belikov, V. V., Ivanov, V. D., Kontorovich, V. K., Korytnik, S. A. and
    Semenov, A. Y. (1997). The non-Sibsonian interpolation: a new method of
    interpolation of the values of a function on an arbitrary set of points.
    *Computational Mathematics and Mathematical Physics*, 37(1), 9-15.

    Examples
    --------
    >>> r = natural_neighbour_weights([(0, 0), (1, 0), (1, 1), (0, 1)], (0.5, 0.5))
    >>> [round(r.weights[i], 12) for i in range(4)]
    [0.25, 0.25, 0.25, 0.25]
    """
    P = _pts(points)
    xx = (float(x[0]), float(x[1]))
    w = _weights(P, xx, method)
    return RichResult(payload={"weights": w, "neighbours": sorted(w)})


def nn_interpolate(points, values, xs, *, method: str = "sibson", outside: str = "nan") -> RichResult:
    r"""Natural-neighbour interpolation ``f(x) = sum_i w_i(x) z_i`` (Sibson or Laplace coordinates).

    Queries outside the convex hull of ``points`` give ``nan``
    (``outside="nan"``) or are extrapolated with the coordinates computed in
    a large enclosing window (``outside="extrapolate"``). Also returns, per
    query, the natural-neighbour variance ``sum w_i (z_i - f)^2`` and the
    bounds ``[min, max]`` of the neighbours' values, between which the
    interpolant always lies (it is a convex combination).

    References
    ----------
    Watson, D. F. (1992). *Contouring: A Guide to the Analysis and Display
    of Spatial Data*. Pergamon.

    Examples
    --------
    >>> P = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 3)]
    >>> r = nn_interpolate(P, [1 + 2 * x - y for x, y in P], [(1.0, 1.0), (1.5, 0.5), (5, 5)])
    >>> [round(v, 10) for v in r.estimates[:2]], r.estimates[2] != r.estimates[2]
    ([2.0, 3.5], True)
    """
    P = _pts(points)
    z = [float(v) for v in values]
    H = _hull(P)
    out, var, lo, hi = [], [], [], []
    for x in xs:
        xx = (float(x[0]), float(x[1]))
        if outside == "nan" and not _in_hull(H, xx):
            out.append(math.nan)
            var.append(math.nan)
            lo.append(math.nan)
            hi.append(math.nan)
            continue
        w = _weights(P, xx, method)
        f = ssum(v * z[i] for i, v in w.items())
        out.append(f)
        var.append(ssum(v * (z[i] - f) ** 2 for i, v in w.items()))
        lo.append(min(z[i] for i in w))
        hi.append(max(z[i] for i in w))
    return RichResult(payload={"estimates": out, "variance": var, "lower": lo, "upper": hi})


def _delaunay_neighbours(P, i):
    """Natural neighbours of data point i among the other data points (Sibson weights of x_i without itself)."""
    Q = [p for j, p in enumerate(P) if j != i]
    idx = [j for j in range(len(P)) if j != i]
    return [idx[k] for k in _weights(Q, P[i], "sibson")]


def nn_gradient_interpolate(points, values, xs, *, method: str = "sibson") -> RichResult:
    r"""Gradient-enhanced natural-neighbour interpolation ``f(x) = sum_i w_i (z_i + g_i . (x - x_i))``.

    The gradient ``g_i`` at each datum is the least-squares plane through
    ``(x_i, z_i)`` and its natural neighbours (the plane is forced through
    the datum and weighted by inverse squared distance); the blend reproduces
    quadratics far better than plain Sibson interpolation and still returns
    ``z_i`` at the data points (Sibson 1981; Watson 1992, chapter 6).
    Outside the convex hull the result is ``nan``.

    Examples
    --------
    >>> P = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 3), (3, 1)]
    >>> r = nn_gradient_interpolate(P, [2 * x + y for x, y in P], [(1.0, 1.0)])
    >>> round(r.estimates[0], 10), [round(v, 10) for v in r.gradients[0]]
    (3.0, [2.0, 1.0])
    """
    P = _pts(points)
    z = [float(v) for v in values]
    grads = []
    for i, p in enumerate(P):
        nb = _delaunay_neighbours(P, i)
        A = [[0.0, 0.0], [0.0, 0.0]]
        rhs = [0.0, 0.0]
        for j in nb:
            dx, dy = P[j][0] - p[0], P[j][1] - p[1]
            wt = 1 / (dx * dx + dy * dy)
            dz = z[j] - z[i]
            A[0][0] += wt * dx * dx
            A[0][1] += wt * dx * dy
            A[1][1] += wt * dy * dy
            rhs[0] += wt * dx * dz
            rhs[1] += wt * dy * dz
        A[1][0] = A[0][1]
        grads.append(solve(A, rhs) if len(nb) >= 2 else [0.0, 0.0])
    H = _hull(P)
    out = []
    for x in xs:
        xx = (float(x[0]), float(x[1]))
        if not _in_hull(H, xx):
            out.append(math.nan)
            continue
        w = _weights(P, xx, method)
        out.append(
            ssum(v * (z[i] + grads[i][0] * (xx[0] - P[i][0]) + grads[i][1] * (xx[1] - P[i][1])) for i, v in w.items())
        )
    return RichResult(payload={"estimates": out, "gradients": [list(g) for g in grads]})


def nn_cross_validation(points, values, *, method: str = "sibson") -> RichResult:
    r"""Leave-one-out cross-validation of natural-neighbour interpolation.

    Each datum is predicted from the others; points on the convex hull of
    the remaining data cannot be interpolated and are skipped (``nan``).
    Returns the predictions, residuals, RMSE and mean error over the
    predicted points.

    Examples
    --------
    >>> P = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]
    >>> r = nn_cross_validation(P, [x + y for x, y in P])
    >>> r.predictions[4], r.rmse
    (2.0, 0.0)
    """
    P = _pts(points)
    z = [float(v) for v in values]
    pred = []
    for i in range(len(P)):
        Q = [p for j, p in enumerate(P) if j != i]
        zq = [v for j, v in enumerate(z) if j != i]
        pred.append(nn_interpolate(Q, zq, [P[i]], method=method).estimates[0])
    res = [p - v for p, v in zip(pred, z)]
    ok = [r for r in res if r == r]
    return RichResult(
        payload={
            "predictions": pred,
            "residuals": res,
            "rmse": math.sqrt(ssum(r * r for r in ok) / len(ok)) if ok else math.nan,
            "mean_error": ssum(ok) / len(ok) if ok else math.nan,
            "n_predicted": len(ok),
        }
    )


def nn_in_domain(points, xs) -> list:
    r"""Whether each query lies in the convex hull of the data -- the natural-neighbour interpolation domain.

    Examples
    --------
    >>> nn_in_domain([(0, 0), (1, 0), (0, 1)], [(0.2, 0.2), (1, 1)])
    [True, False]
    """
    H = _hull(_pts(points))
    return [_in_hull(H, (float(x[0]), float(x[1]))) for x in xs]


def cheatsheet() -> str:
    return (
        "natural_neighbour_weights / nn_interpolate / nn_gradient_interpolate / nn_cross_validation / "
        "nn_in_domain -> Sibson and Laplace natural-neighbour interpolation."
    )
