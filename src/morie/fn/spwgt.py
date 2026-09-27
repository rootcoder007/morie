"""Spatial weight matrix construction: neighbour graphs, distance weights, coding styles."""

from __future__ import annotations

import math

from . import _array_core as np
from ._containers import SpatialResult
from ._qpcore import ssum

_STYLES = ("B", "W", "C", "U", "S", "minmax")
_KERNELS = ("uniform", "triangular", "epanechnikov", "quartic", "gaussian")


def _kernel(z, name):
    if name == "gaussian":
        return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    if z > 1:
        return 0.0
    if name == "uniform":
        return 0.5
    if name == "triangular":
        return 1 - z
    if name == "epanechnikov":
        return 0.75 * (1 - z * z)
    return 15.0 / 16.0 * (1 - z * z) ** 2


def spatial_weights(
    coords,
    method: str = "knn",
    k: int = 5,
    threshold: float | None = None,
    row_standardize: bool = True,
    *,
    style: str | None = None,
    d_min: float = 0.0,
    alpha: float = 1.0,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> SpatialResult:
    r"""Construct a spatial weight matrix from point coordinates.

    Neighbour graphs (binary before coding):

    - ``knn``: the ``k`` nearest other points (ties broken by index), as
      ``spdep::knn2nb(knearneigh(x, k))``; not symmetric in general.
    - ``distance``: ``d_min < d_ij <= threshold`` (``spdep::dnearneigh``);
      ``threshold`` defaults to the median inter-point distance.
    - ``gabriel``: no third point in the circle with diameter ``ij``
      (``d_ik^2 + d_jk^2 < d_ij^2`` for no ``k``; ``spdep::gabrielneigh``).
    - ``relative``: no third point closer to both
      (``max(d_ik, d_jk) < d_ij`` for no ``k``; ``spdep::relativeneigh``).
    - ``delaunay``: the Delaunay edges (Bowyer-Watson).

    Valued weights: ``inverse`` gives ``d_ij^(-alpha)`` within the distance
    band; ``kernel`` gives ``K(d_ij / bandwidth)`` for ``d_ij > 0`` with the
    uniform, triangular, Epanechnikov, quartic or Gaussian kernel (zero
    beyond the bandwidth except Gaussian); ``bandwidth`` defaults to the
    ``k``-th nearest-neighbour distance of each point (adaptive).

    Coding styles of ``spdep::nb2listw``: ``B`` (as built), ``W`` (row
    sums 1), ``C`` (globally scaled to sum to the number of non-island
    units), ``U`` (sum 1), ``S`` (Tiefelsdorf, Griffith and Boots 1999: rows
    divided by their Euclidean norm, then scaled to sum to ``n``) and
    ``minmax`` (divided by the smaller of the largest row and column
    sums). ``style`` overrides ``row_standardize`` (``W`` if true, else
    ``B``). Self-weights are zero.

    :param coords: (n, 2) coordinates.
    :param method: ``knn``, ``distance``, ``inverse``, ``kernel``,
        ``gabriel``, ``relative`` or ``delaunay``.
    :param k: Neighbours for ``knn`` and the adaptive bandwidth.
    :param threshold: Upper distance for ``distance`` and ``inverse``.
    :param row_standardize: Row-standardise when ``style`` is not given.
    :param style: Coding style (see above).
    :param d_min: Lower distance for ``distance`` and ``inverse``.
    :param alpha: Power for ``inverse``.
    :param bandwidth: Kernel bandwidth (default adaptive).
    :param kernel: Kernel for ``kernel``.
    :return: SpatialResult; ``statistic`` is the mean number of neighbours;
        ``extra`` has ``W``, ``neighbours``, ``style``, ``n_islands``,
        ``components`` (label per unit), ``n_components``, ``sparsity``
        (share of non-zero off-diagonal entries), ``S0``, ``S1``, ``S2``,
        ``symmetric``, ``method``, ``n``, ``row_standardized``.

    References
    ----------
    Anselin L (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
    Chapter 3.

    Gabriel K R, Sokal R R (1969). A new statistical approach to geographic
    variation analysis. *Systematic Zoology* 18, 259-278.

    Toussaint G T (1980). The relative neighbourhood graph of a finite
    planar set. *Pattern Recognition* 12, 261-268.

    Tiefelsdorf M, Griffith D A, Boots B (1999). A variance-stabilizing
    coding scheme for spatial link matrices. *Environment and Planning A*
    31, 165-180.

    Getis A, Aldstadt J (2004). Constructing the spatial weights matrix
    using a local statistic. *Geographical Analysis*, 36(2), 90--104.
    doi:10.1111/j.1538-4632.2004.tb01127.x

    Examples
    --------
    >>> coords = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]]
    >>> res = spatial_weights(coords, method="gabriel", style="B")
    >>> [sorted(v) for v in res.extra["neighbours"]]
    [[1, 2, 4], [0, 3, 4], [0, 3, 4], [1, 2, 4], [0, 1, 2, 3]]
    """
    P = [[float(v) for v in p] for p in np.asarray(coords, dtype=float).tolist()]
    if any(len(p) != 2 for p in P):
        raise ValueError("coords must be (n, 2).")
    n = len(P)
    D = [[math.dist(a, b) for b in P] for a in P]
    if style is None:
        style = "W" if row_standardize else "B"
    if style not in _STYLES:
        raise ValueError(f"style must be one of {_STYLES}")
    G = [[0.0] * n for _ in range(n)]
    if method in ("distance", "inverse") and threshold is None:
        threshold = float(sorted(D[i][j] for i in range(n) for j in range(i + 1, n))[(n * (n - 1) // 2 - 1) // 2])
    if method == "knn":
        for i in range(n):
            for j in sorted((j for j in range(n) if j != i), key=lambda j: (D[i][j], j))[:k]:
                G[i][j] = 1.0
    elif method == "distance":
        for i in range(n):
            for j in range(n):
                if i != j and d_min < D[i][j] <= threshold:
                    G[i][j] = 1.0
    elif method == "inverse":
        for i in range(n):
            for j in range(n):
                if i != j and d_min < D[i][j] <= threshold:
                    G[i][j] = D[i][j] ** (-alpha)
    elif method == "kernel":
        if kernel not in _KERNELS:
            raise ValueError(f"kernel must be one of {_KERNELS}")
        for i in range(n):
            h = bandwidth if bandwidth is not None else sorted(D[i][j] for j in range(n) if j != i)[k - 1]
            for j in range(n):
                if i != j and D[i][j] > 0:
                    G[i][j] = _kernel(D[i][j] / h, kernel)
    elif method in ("gabriel", "relative"):
        for i in range(n):
            for j in range(i + 1, n):
                dij = D[i][j]
                if method == "gabriel":
                    blocked = any(D[i][m] ** 2 + D[j][m] ** 2 < dij**2 for m in range(n) if m not in (i, j))
                else:
                    blocked = any(max(D[i][m], D[j][m]) < dij for m in range(n) if m not in (i, j))
                if not blocked:
                    G[i][j] = G[j][i] = 1.0
    elif method == "delaunay":
        from .deltri import delaunay_triangulation

        for i, j in delaunay_triangulation(P).extra["edges"]:
            G[i][j] = G[j][i] = 1.0
    else:
        raise ValueError(f"Unknown method '{method}'.")
    rows = [ssum(r) for r in G]
    islands = [i for i in range(n) if not any(G[i])]
    eff = n - len(islands)
    if style == "W":
        W = [[v / rows[i] if rows[i] > 0 else 0.0 for v in G[i]] for i in range(n)]
    elif style in ("C", "U", "minmax"):
        tot = ssum(rows)
        f = eff / tot if style == "C" else 1.0 / tot if style == "U" else 1.0
        W = [[v * f for v in r] for r in G]
        if style == "minmax":
            cols = [ssum(G[i][j] for i in range(n)) for j in range(n)]
            mm = min(max(rows), max(cols))
            W = [[v / mm for v in r] for r in G]
    elif style == "S":
        q = [math.sqrt(ssum(v * v for v in r)) for r in G]
        V = [[v / q[i] if q[i] > 0 else 0.0 for v in G[i]] for i in range(n)]
        Q = ssum(ssum(r) for r in V)
        W = [[v * n / Q for v in r] for r in V]
    else:
        W = [list(r) for r in G]
    nbrs = [[j for j in range(n) if W[i][j] != 0.0] for i in range(n)]
    label, comp = [-1] * n, 0
    for s in range(n):
        if label[s] >= 0:
            continue
        stack, label[s] = [s], comp
        while stack:
            a = stack.pop()
            for b in range(n):
                if (G[a][b] or G[b][a]) and label[b] < 0:
                    label[b] = comp
                    stack.append(b)
        comp += 1
    S0 = ssum(ssum(r) for r in W)
    S1 = 0.5 * ssum((W[i][j] + W[j][i]) ** 2 for i in range(n) for j in range(n))
    S2 = ssum((ssum(W[i]) + ssum(W[j][i] for j in range(n))) ** 2 for i in range(n))
    return SpatialResult(
        name="spatial_weights",
        statistic=float(ssum(len(v) for v in nbrs) / n),
        extra={
            "W": np.asarray(W, dtype=float),
            "neighbours": nbrs,
            "style": style,
            "n_islands": len(islands),
            "components": label,
            "n_components": comp,
            "sparsity": ssum(len(v) for v in nbrs) / (n * (n - 1)) if n > 1 else 0.0,
            "S0": S0,
            "S1": S1,
            "S2": S2,
            "symmetric": all(W[i][j] == W[j][i] for i in range(n) for j in range(i)),
            "method": method,
            "n": n,
            "row_standardized": style == "W",
        },
    )


spwgt = spatial_weights


def cheatsheet() -> str:
    return "spatial_weights(coords) -> neighbour graphs, distance and kernel weights, spdep coding styles."


# compact alias per ledger/NAMING.md
spatialweights = spatial_weights
