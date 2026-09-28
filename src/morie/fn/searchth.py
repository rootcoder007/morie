# morie.fn -- function file (rootcoder007/morie)
"""Search theory for search and rescue planning (Koopman; USCG "The Theory of Search", Frost 1996): lateral range
models and sweep width, coverage and probability of detection, POC/POD/POS bookkeeping and Bayesian updating,
datum probability maps, and optimal allocation of search effort."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "lateral_range",
    "sweep_width",
    "search_pod",
    "parallel_sweep_pod",
    "search_success",
    "datum_probability_map",
    "optimal_circular_search",
    "optimal_effort_allocation",
]

_SQRT2 = math.sqrt(2.0)


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _phi(z):
    return 0.5 * math.erfc(-z / _SQRT2)


def lateral_range(x, W: float, *, model: str = "inverse_cube", m: float = 1.0):
    r"""Lateral range curve ``p(x)``: probability of detecting an object passed at lateral range ``x``.

    - ``definite``: ``p = 1`` for ``|x| <= W/2``, else 0;
    - ``mbeta``: ``p = m`` for ``|x| <= W/(2m)``, else 0 (the M-Beta model;
      ``m = 1`` is the definite range model);
    - ``inverse_cube``: ``p = 1 - exp(-W^2 / (4 pi x^2))`` (Koopman's law;
      Frost 1996, eq. 4-13).

    Each has sweep width (area under the curve) ``W``.

    References
    ----------
    Koopman, B. O. (1946). *Search and Screening*. OEG Report 56, Office of
    the Chief of Naval Operations, Washington.
    Frost, J. R. (1996). *The Theory of Search: A Simplified Explanation*.
    Soza & Company and U.S. Coast Guard Office of Search and Rescue.

    Examples
    --------
    >>> [round(v, 6) for v in lateral_range([0.5, 1.0, 2.0], 1.0)]
    [0.272623, 0.076494, 0.019698]
    """
    out = []
    for v in _vec(x):
        a = abs(v)
        if model == "definite":
            out.append(1.0 if a <= W / 2 else 0.0)
        elif model == "mbeta":
            out.append(m if a <= W / (2 * m) else 0.0)
        elif model == "inverse_cube":
            out.append(1.0 if a == 0 else 1 - math.exp(-W * W / (4 * math.pi * a * a)))
        else:
            raise ValueError("model must be definite, mbeta or inverse_cube")
    return out


def sweep_width(x, p) -> float:
    r"""Sweep (effective search) width ``W = int p(x) dx`` from a sampled lateral range curve (trapezoidal rule).

    Examples
    --------
    >>> sweep_width([-2, -1, 0, 1, 2], [0, 1, 1, 1, 0])
    3.0
    """
    X, P = _vec(x), _vec(p)
    return ssum((X[i + 1] - X[i]) * (P[i] + P[i + 1]) / 2 for i in range(len(X) - 1))


def search_pod(coverage, *, model: str = "random") -> list:
    r"""Probability of detection as a function of coverage ``C = W z / A = W / S``.

    ``random`` (Koopman's random search) ``1 - exp(-C)``; parallel sweeps
    with a ``definite`` range sensor ``min(C, 1)``; ``inverse_cube``
    ``erf(sqrt(pi) C / 2)`` (Frost 1996, eqs. 5-3, 5-13).

    Examples
    --------
    >>> [round(v, 4) for v in search_pod([1.0], model="inverse_cube")]
    [0.7899]
    """
    out = []
    for c in _vec(coverage):
        if model == "random":
            out.append(1 - math.exp(-c))
        elif model == "definite":
            out.append(min(c, 1.0))
        elif model == "inverse_cube":
            out.append(math.erf(math.sqrt(math.pi) * c / 2))
        else:
            raise ValueError("model must be random, definite or inverse_cube")
    return out


def parallel_sweep_pod(
    W: float,
    S: float,
    *,
    model: str = "inverse_cube",
    m: float = 1.0,
    nav_sd: float = 0.0,
    n_points: int = 2001,
    n_tracks: int = 60,
) -> float:
    r"""POD of a parallel-sweep search with track spacing ``S`` for any lateral range model, by numerical averaging.

    With tracks at ``kS`` (``k = -n_tracks..n_tracks``) and an object uniform
    across one spacing, ``POD = (1/S) int_0^S [1 - prod_k (1 - p(x - kS))] dx``
    (composite Simpson rule on ``n_points``); for the inverse cube law the
    tracks beyond ``n_tracks`` are added in closed form through ``sum_k
    1/(x - kS)^2 = (pi/S)^2 / sin^2(pi x/S)``, so without navigation error
    the result is Koopman's ``erf(sqrt(pi) W / (2 S))``. ``nav_sd > 0`` replaces ``p``
    by its convolution with a normal track error of that standard deviation
    (Gauss-Hermite, 40 nodes), the navigational-error effect that pushes PODs
    toward the random-search curve (Frost 1996, 5.10-5.12).

    Examples
    --------
    >>> round(parallel_sweep_pod(1.0, 1.0), 4)
    0.7899
    """
    if n_points % 2 == 0:
        n_points += 1
    if nav_sd > 0:
        gh = _gauss_hermite(40)

        def p(x):
            return ssum(
                w * lateral_range([x + _SQRT2 * nav_sd * t], W, model=model, m=m)[0] for t, w in gh
            ) / math.sqrt(math.pi)
    else:

        def p(x):
            return lateral_range([x], W, model=model, m=m)[0]

    h = S / (n_points - 1)
    vals = []
    for i in range(n_points):
        x = i * h
        miss = 1.0
        for k in range(-n_tracks, n_tracks + 1):
            miss *= 1 - p(x - k * S)
        if model == "inverse_cube" and miss > 0:
            # tracks beyond n_tracks: p ~ W^2/(4 pi d^2); sum_k 1/(x - kS)^2 = (pi/S)^2 / sin^2(pi x / S)
            near = ssum(1 / (x - k * S) ** 2 for k in range(-n_tracks, n_tracks + 1) if x != k * S)
            full = (math.pi / S) ** 2 / math.sin(math.pi * x / S) ** 2 if math.sin(math.pi * x / S) != 0 else near
            miss *= math.exp(-W * W / (4 * math.pi) * max(0.0, full - near))
        vals.append(1 - miss)
    simpson = vals[0] + vals[-1] + 4 * ssum(vals[1:-1:2]) + 2 * ssum(vals[2:-1:2])
    return simpson * h / 3 / S


def _gauss_hermite(n):
    """Nodes and weights of n-point Gauss-Hermite quadrature (Golub-Welsch via symmetric tridiagonal eigenproblem)."""
    J = [[0.0] * n for _ in range(n)]
    for i in range(1, n):
        b = math.sqrt(i / 2)
        J[i][i - 1] = J[i - 1][i] = b
    vals, vecs = np.linalg.eigh(np.asarray(J, dtype=float))
    V = [[float(v) for v in r] for r in vecs.tolist()]
    return [(float(vals[k]), math.sqrt(math.pi) * V[0][k] ** 2) for k in range(n)]


def search_success(poc, pod) -> RichResult:
    r"""Search bookkeeping after one unsuccessful search of each area (Frost 1996, chapter 2).

    ``POS_i = POC_i POD_i``, total ``POS = sum POS_i``; the Bayes-updated
    containment ``POC_i' = POC_i (1 - POD_i) / (1 - POS)`` (the adjusted POC,
    2.8), and, given a list of successive PODs per area (rows = searches),
    the cumulative ``POD_c = 1 - prod (1 - POD_k)`` and the cumulative
    ``POS = sum_i POC_i POD_c,i`` over the original POCs (2.5, 2.9), with the
    POCs updated after every search.

    Examples
    --------
    >>> r = search_success([0.5, 0.3, 0.2], [0.8, 0.5, 0.0])
    >>> round(r.pos, 6), [round(v, 6) for v in r.poc_updated]
    (0.55, [0.222222, 0.333333, 0.444444])
    """
    P = _vec(poc)
    D = [_vec(r) for r in (pod if isinstance(pod[0], (list, tuple)) else [pod])]
    cur = P
    for r in D:  # Bayes update after each unsuccessful search
        pos = ssum(a * b for a, b in zip(cur, r))
        cur = [a * (1 - b) / (1 - pos) for a, b in zip(cur, r)]
    cum_pod = [1 - math.prod(1 - r[i] for r in D) for i in range(len(P))]
    by_area = [a * b for a, b in zip(P, cum_pod)]
    return RichResult(
        payload={"pos": ssum(by_area), "pos_by_area": by_area, "poc_updated": cur, "cumulative_pod": cum_pod}
    )


def datum_probability_map(
    x_edges, y_edges, *, datum: str = "point", center=(0.0, 0.0), sigma: float = 1.0, end=None, nodes: int = 64
) -> RichResult:
    r"""Probability of containment of grid cells for point, line and area datums (Frost 1996, chapter 3).

    - ``point``: circular normal about ``center`` with standard deviation
      ``sigma`` per axis: cell probability ``[Phi(x2) - Phi(x1)][Phi(y2) -
      Phi(y1)]``;
    - ``line``: position uniform along the segment ``center``-``end`` plus a
      circular normal error (integrated along the segment with ``nodes``
      Gauss-Legendre points);
    - ``area``: uniform over the rectangle with corners ``center`` and ``end``.

    Also returned: the probability outside the grid.

    Examples
    --------
    >>> r = datum_probability_map([-1, 0, 1], [-1, 0, 1])
    >>> round(r.poc[0][0], 6), round(r.outside, 6)
    (0.116516, 0.533935)
    """
    xe, ye = _vec(x_edges), _vec(y_edges)
    cx, cy = float(center[0]), float(center[1])

    def point_cell(px, py, i, j):
        return (_phi((xe[i + 1] - px) / sigma) - _phi((xe[i] - px) / sigma)) * (
            _phi((ye[j + 1] - py) / sigma) - _phi((ye[j] - py) / sigma)
        )

    grid = [[0.0] * (len(xe) - 1) for _ in range(len(ye) - 1)]
    if datum == "point":
        for j in range(len(ye) - 1):
            for i in range(len(xe) - 1):
                grid[j][i] = point_cell(cx, cy, i, j)
    elif datum == "line":
        ex, ey = float(end[0]), float(end[1])
        gl = _gauss_legendre(nodes)
        for j in range(len(ye) - 1):
            for i in range(len(xe) - 1):
                grid[j][i] = ssum(
                    w / 2 * point_cell(cx + (t + 1) / 2 * (ex - cx), cy + (t + 1) / 2 * (ey - cy), i, j) for t, w in gl
                )
    elif datum == "area":
        x1, x2 = sorted((cx, float(end[0])))
        y1, y2 = sorted((cy, float(end[1])))
        A = (x2 - x1) * (y2 - y1)
        for j in range(len(ye) - 1):
            for i in range(len(xe) - 1):
                ox = max(0.0, min(x2, xe[i + 1]) - max(x1, xe[i]))
                oy = max(0.0, min(y2, ye[j + 1]) - max(y1, ye[j]))
                grid[j][i] = ox * oy / A
    else:
        raise ValueError("datum must be point, line or area")
    tot = ssum(ssum(r) for r in grid)
    return RichResult(payload={"poc": grid, "outside": 1 - tot})


def _gauss_legendre(n):
    J = [[0.0] * n for _ in range(n)]
    for i in range(1, n):
        b = i / math.sqrt(4 * i * i - 1)
        J[i][i - 1] = J[i - 1][i] = b
    vals, vecs = np.linalg.eigh(np.asarray(J, dtype=float))
    V = [[float(v) for v in r] for r in vecs.tolist()]
    return [(float(vals[k]), 2 * V[0][k] ** 2) for k in range(n)]


def optimal_circular_search(sigma: float, W: float, effort: float) -> RichResult:
    r"""Optimal effort density for a circular normal datum under random-search detection (Koopman 1946).

    Maximising ``int p(r) (1 - exp(-W z(r))) dA`` subject to ``int z dA = Z``
    gives ``z(r) = (R^2 - r^2)/(2 sigma^2 W)`` inside the optimal radius ``R =
    (4 sigma^2 W Z / pi)^{1/4}`` and 0 outside; the probability of success is
    ``1 - (1 + u) exp(-u)``, ``u = R^2 / (2 sigma^2)``.

    References
    ----------
    Koopman, B. O. (1946). *Search and Screening*. OEG Report 56.
    Stone, L. D. (1975). *Theory of Optimal Search*. Academic Press.

    Examples
    --------
    >>> r = optimal_circular_search(1.0, 1.0, math.pi / 4)
    >>> round(r.radius, 6), round(r.pos, 6)
    (1.0, 0.090204)
    """
    R = (4 * sigma * sigma * W * effort / math.pi) ** 0.25
    u = R * R / (2 * sigma * sigma)
    return RichResult(
        payload={"radius": R, "pos": 1 - (1 + u) * math.exp(-u), "center_density": R * R / (2 * sigma * sigma * W)}
    )


def optimal_effort_allocation(poc, area, W: float, effort: float, *, tol: float = 1e-14) -> RichResult:
    r"""Optimal allocation of search effort (track length) over cells for random-search detection (Stone 1975).

    Maximises ``sum_i POC_i (1 - exp(-W z_i / A_i))`` subject to ``sum z_i =
    Z``: ``z_i = (A_i / W) ln(POC_i W / (A_i lambda))`` on cells with ``POC_i
    / A_i > lambda / W`` (the equalisation of posterior densities -- the
    additive principle of Frost 1996, 6.6), ``lambda`` found by bisection on
    ``log lambda``. Returns efforts, coverages, cell PODs and the total POS.

    Examples
    --------
    >>> r = optimal_effort_allocation([0.5, 0.5], [1.0, 1.0], 1.0, 2.0)
    >>> [round(v, 6) for v in r.effort], round(r.pos, 6)
    ([1.0, 1.0], 0.632121)
    """
    P, A = _vec(poc), _vec(area)

    def total(lam):
        return ssum(a / W * math.log(p * W / (a * lam)) for p, a in zip(P, A) if p * W / (a * lam) > 1)

    lo = math.log(min(p * W / a for p, a in zip(P, A) if p > 0)) - 50
    hi = math.log(max(p * W / a for p, a in zip(P, A)))
    for _ in range(400):
        mid = (lo + hi) / 2
        if total(math.exp(mid)) > effort:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    lam = math.exp((lo + hi) / 2)
    z = [a / W * math.log(p * W / (a * lam)) if p * W / (a * lam) > 1 else 0.0 for p, a in zip(P, A)]
    cov = [W * zi / a for zi, a in zip(z, A)]
    pod = [1 - math.exp(-c) for c in cov]
    return RichResult(
        payload={"effort": z, "coverage": cov, "pod": pod, "pos": ssum(p * d for p, d in zip(P, pod)), "lambda": lam}
    )


def cheatsheet() -> str:
    return "lateral_range / search_pod / search_success / datum_probability_map / optimal_effort_allocation -> SAR."
