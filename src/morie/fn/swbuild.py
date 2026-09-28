# morie.fn -- function file (rootcoder007/morie)
"""Spatial weights construction: grid contiguity (queen, rook), distance bands, k nearest neighbours, inverse
distance, fixed and adaptive kernels, symmetrisation and connected components."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "grid_contiguity",
    "distance_band_weights",
    "knn_neighbour_weights",
    "inverse_distance_weights",
    "kernel_weights",
    "symmetrize_weights",
    "weights_components",
]


def _pts(coords):
    return [tuple(float(v) for v in p) for p in coords]


def _d(p, q):
    # sqrt of summed squares, as R dist() and spdep, so band edges agree across arms
    return math.sqrt(ssum((a - b) ** 2 for a, b in zip(p, q)))


def grid_contiguity(nrow: int, ncol: int, *, type: str = "queen", torus: bool = False):
    r"""Binary contiguity of a regular ``nrow`` x ``ncol`` grid, cells numbered row by row (``spdep::cell2nb``).

    ``rook`` joins cells sharing an edge, ``queen`` also those sharing a
    corner; ``torus`` wraps the edges.

    Examples
    --------
    >>> W = grid_contiguity(2, 2, type="rook")
    >>> W[0]
    [0.0, 1.0, 1.0, 0.0]
    """
    if type not in ("queen", "rook"):
        raise ValueError("type must be queen or rook")
    n = nrow * ncol
    W = [[0.0] * n for _ in range(n)]
    for r in range(nrow):
        for c in range(ncol):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr == 0 and dc == 0) or (type == "rook" and dr and dc):
                        continue
                    rr, cc = r + dr, c + dc
                    if torus:
                        rr, cc = rr % nrow, cc % ncol
                    elif not (0 <= rr < nrow and 0 <= cc < ncol):
                        continue
                    j = rr * ncol + cc
                    if j != r * ncol + c:
                        W[r * ncol + c][j] = 1.0
    return W


def distance_band_weights(coords, d2: float, *, d1: float = 0.0):
    r"""Binary distance-band neighbours ``d1 < d_ij <= d2`` (``spdep::dnearneigh`` bounds ``GT``, ``LE``).

    Examples
    --------
    >>> distance_band_weights([(0, 0), (1, 0), (3, 0)], 1.5)[0]
    [0.0, 1.0, 0.0]
    """
    P = _pts(coords)
    return [[1.0 if i != j and d1 < _d(p, q) <= d2 else 0.0 for j, q in enumerate(P)] for i, p in enumerate(P)]


def knn_neighbour_weights(coords, k: int):
    r"""Binary ``k`` nearest neighbours (``spdep::knearneigh`` + ``knn2nb``), ties broken by the lower index.

    The result is generally asymmetric; see :func:`symmetrize_weights`.

    Examples
    --------
    >>> knn_neighbour_weights([(0, 0), (1, 0), (3, 0)], 1)
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    """
    P = _pts(coords)
    n = len(P)
    if not 0 < k < n:
        raise ValueError("k must be in 1..n-1")
    W = [[0.0] * n for _ in range(n)]
    for i, p in enumerate(P):
        order = sorted((j for j in range(n) if j != i), key=lambda j: (_d(p, P[j]), j))
        for j in order[:k]:
            W[i][j] = 1.0
    return W


def inverse_distance_weights(coords, *, power: float = 1.0, d2: float | None = None, row_standardize: bool = False):
    r"""Inverse-distance weights ``w_ij = d_ij^{-power}`` (within ``d2`` if given), optionally row-standardised.

    Examples
    --------
    >>> inverse_distance_weights([(0, 0), (2, 0)], power=2.0)
    [[0.0, 0.25], [0.25, 0.0]]
    """
    P = _pts(coords)
    W = []
    for i, p in enumerate(P):
        row = []
        for j, q in enumerate(P):
            d = _d(p, q)
            row.append(0.0 if i == j or d == 0 or (d2 is not None and d > d2) else d ** (-power))
        if row_standardize:
            s = ssum(row)
            row = [v / s if s > 0 else 0.0 for v in row]
        W.append(row)
    return W


def _kern(d, bw, kernel):
    u = d / bw
    if kernel == "gaussian":
        return math.exp(-0.5 * u * u)
    if kernel == "exponential":
        return math.exp(-u)
    if kernel == "bisquare":
        return (1 - u * u) ** 2 if d < bw else 0.0
    if kernel == "tricube":
        return (1 - u**3) ** 3 if d < bw else 0.0
    if kernel == "boxcar":
        return 1.0 if d <= bw else 0.0
    raise ValueError("unknown kernel")


def kernel_weights(coords, bw: float, *, kernel: str = "bisquare", adaptive: bool = False, diagonal: bool = True):
    r"""Geographically weighted kernel weights, as ``GWmodel::gw.weight``.

    Fixed bandwidth ``bw`` (a distance) or ``adaptive`` (``bw`` = number of
    nearest points, the bandwidth of row ``i`` being the distance to its
    ``bw``-th nearest point counting itself). Kernels with ``u = d/b``:
    gaussian ``exp(-u^2/2)``, exponential ``exp(-u)``, bisquare ``(1 -
    u^2)^2``, tricube ``(1 - u^3)^3`` (both zero for ``d >= b``) and boxcar
    ``1(d <= b)``. ``diagonal=False`` zeroes ``w_ii`` for use as spatial
    weights.

    References
    ----------
    Gollini, I., Lu, B., Charlton, M., Brunsdon, C. and Harris, P. (2015).
    GWmodel: an R package for exploring spatial heterogeneity using
    geographically weighted models. *Journal of Statistical Software*,
    63(17), 1-50.

    Examples
    --------
    >>> [round(v, 6) for v in kernel_weights([(0, 0), (1, 0), (3, 0)], 2.0)[0]]
    [1.0, 0.5625, 0.0]
    """
    P = _pts(coords)
    n = len(P)
    if adaptive and not 1 <= int(bw) <= n:
        raise ValueError("adaptive bw must be between 1 and n")
    out = []
    for i, p in enumerate(P):
        d = [_d(p, q) for q in P]
        b = sorted(d)[int(bw) - 1] if adaptive else float(bw)
        row = [_kern(dj, b, kernel) for dj in d]
        if not diagonal:
            row[i] = 0.0
        out.append(row)
    return out


def symmetrize_weights(W, *, method: str = "union"):
    r"""Symmetrise a weights matrix: ``union`` (``1`` if either link exists, ``spdep::make.sym.nb``), ``intersection``, or ``average`` ``(W + W')/2``.

    Examples
    --------
    >>> symmetrize_weights([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    """
    A = [[float(v) for v in r] for r in W]
    n = len(A)
    if method == "union":
        return [[1.0 if A[i][j] or A[j][i] else 0.0 for j in range(n)] for i in range(n)]
    if method == "intersection":
        return [[1.0 if A[i][j] and A[j][i] else 0.0 for j in range(n)] for i in range(n)]
    if method == "average":
        return [[(A[i][j] + A[j][i]) / 2 for j in range(n)] for i in range(n)]
    raise ValueError("method must be union, intersection or average")


def weights_components(W) -> RichResult:
    r"""Connected components of the (undirected) neighbour graph, as ``spdep::n.comp.nb``.

    Components are numbered 1, 2, ... in order of their lowest-index unit.

    Examples
    --------
    >>> r = weights_components([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    >>> r.n_components, r.component
    (2, [1, 1, 2])
    """
    A = [[float(v) for v in r] for r in W]
    n = len(A)
    comp = [0] * n
    c = 0
    for s in range(n):
        if comp[s]:
            continue
        c += 1
        comp[s] = c
        stack = [s]
        while stack:
            i = stack.pop()
            for j in range(n):
                if (A[i][j] or A[j][i]) and not comp[j]:
                    comp[j] = c
                    stack.append(j)
    return RichResult(
        payload={
            "n_components": c,
            "component": comp,
            "connected": c == 1,
            "sizes": [comp.count(k) for k in range(1, c + 1)],
        }
    )


def cheatsheet() -> str:
    return "grid_contiguity / distance_band_weights / knn_neighbour_weights / kernel_weights -> spatial weights construction."
