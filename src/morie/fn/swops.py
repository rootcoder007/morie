# morie.fn -- function file (rootcoder007/morie)
"""Spatial weight operators: rho bounds, lag and error operators, block, regime and contiguity weights."""

from __future__ import annotations

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult

__all__ = [
    "rho_bounds",
    "lag_operator",
    "error_operator",
    "block_weights",
    "regime_weights",
    "polygon_contiguity",
    "neighbour_cardinality",
    "compare_neighbours",
]


def _mat(W):
    return [[float(v) for v in r] for r in np.asarray(W, dtype=float).tolist()]


def _square(W):
    n = len(W)
    if any(len(r) != n for r in W):
        raise ValueError("W must be square")
    return n


def rho_bounds(W, tol: float = 1e-10) -> RichResult:
    r"""Eigenvalue bounds on the autoregressive parameter of ``I - rho W``.

    ``I - rho W`` is singular exactly at ``rho = 1/omega`` for the
    eigenvalues ``omega`` of ``W``; the interval containing 0 on which it
    stays invertible is ``(1/omega_min, 1/omega_max)`` (Ord 1975;
    ``spatialreg::eigenw`` and the ``interval`` of ``spatialreg::lagsarlm``),
    and the Neumann series ``sum_k rho^k W^k`` converges when
    ``|rho| < 1/spectral radius``.  The eigenvalues are real when ``W`` is
    symmetric or similar to a symmetric matrix, as a row-standardised
    symmetric neighbour matrix ``D^{-1}B`` is (via ``D^{-1/2} B D^{-1/2}``);
    other ``W`` are rejected.

    :param W: Spatial weights (n, n).
    :param tol: Tolerance for the symmetry checks.
    :return: :class:`RichResult` with ``lower``, ``upper``,
        ``spectral_radius``, ``eigenvalues`` (increasing) and
        ``neumann_radius`` (``1/spectral_radius``).
    :raises ValueError: If ``W`` is not symmetric nor ``D^{-1}B`` with
        symmetric ``B``.

    References
    ----------
    Ord, K. (1975). Estimation methods for models of spatial interaction.
    *Journal of the American Statistical Association*, 70(349), 120-126.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> r = rho_bounds(W)
    >>> round(r.lower, 6), round(r.upper, 6)
    (-1.0, 1.0)
    """
    W = _mat(W)
    n = _square(W)
    sym = all(abs(W[i][j] - W[j][i]) <= tol for i in range(n) for j in range(n))
    if sym:
        Ssym = W
    else:
        # W = D^{-1} B: recover B = diag(d) W with d_i chosen so B is symmetric
        B = [[1.0 if W[i][j] != 0.0 else 0.0 for j in range(n)] for i in range(n)]
        d = [sum(r) for r in B]
        ok = all(abs(W[i][j] - (B[i][j] / d[i] if d[i] else 0.0)) <= tol for i in range(n) for j in range(n))
        ok = ok and all(B[i][j] == B[j][i] for i in range(n) for j in range(n))
        if not ok:
            raise ValueError("W must be symmetric or a row-standardised symmetric neighbour matrix")
        Ssym = [[B[i][j] / (d[i] * d[j]) ** 0.5 if d[i] and d[j] else 0.0 for j in range(n)] for i in range(n)]
    w, _v = np.linalg.eigh(np.asarray(Ssym))
    ev = sorted(float(v) for v in w)
    lo, hi = ev[0], ev[-1]
    sr = max(abs(lo), abs(hi))
    return RichResult(
        payload={
            "lower": 1.0 / lo if lo < 0 else float("-inf"),
            "upper": 1.0 / hi if hi > 0 else float("inf"),
            "spectral_radius": sr,
            "neumann_radius": 1.0 / sr if sr > 0 else float("inf"),
            "eigenvalues": ev,
        }
    )


def lag_operator(W, x, power: int = 1):
    """Spatial lag ``W^k x`` (``spdep::lag.listw`` applied ``k`` times).

    :param W: Spatial weights (n, n).
    :param x: Values (n,).
    :param power: Order ``k >= 0``.
    :return: List of the ``n`` lagged values.

    Examples
    --------
    >>> lag_operator([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]], [1.0, 2.0, 4.0])
    [2.0, 2.5, 2.0]
    """
    W = _mat(W)
    n = _square(W)
    v = [float(a) for a in np.asarray(x, dtype=float).tolist()]
    if len(v) != n:
        raise ValueError("x must have length n")
    if int(power) < 0:
        raise ValueError("power must be non-negative")
    for _ in range(int(power)):
        v = [ssum(W[i][j] * v[j] for j in range(n)) for i in range(n)]
    return v


def error_operator(W, rho: float):
    """The spatial multiplier ``(I - rho W)^{-1}`` (``spatialreg::invIrW``).

    It maps innovations to the autoregressive process ``u = rho W u + e``.

    :param W: Spatial weights (n, n).
    :param rho: Autoregressive parameter; ``I - rho W`` must be invertible.
    :return: (n, n) list of lists.

    Examples
    --------
    >>> M = error_operator([[0, 1], [1, 0]], 0.5)
    >>> [[round(v, 6) for v in r] for r in M]
    [[1.333333, 0.666667], [0.666667, 1.333333]]
    """
    W = _mat(W)
    n = _square(W)
    A = [[(1.0 if i == j else 0.0) - rho * W[i][j] for j in range(n)] for i in range(n)]
    return [[float(v) for v in r] for r in inverse(A)]


def block_weights(groups):
    """Binary weights linking every pair of distinct units in the same group.

    As ``spdep::nb2blocknb`` with no prior neighbours: ``w_ij = 1`` when
    ``i != j`` share a group label.

    Examples
    --------
    >>> block_weights(["a", "b", "a"])
    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    """
    g = list(groups)
    n = len(g)
    return [[1.0 if i != j and g[i] == g[j] else 0.0 for j in range(n)] for i in range(n)]


def regime_weights(W, regimes, row_standardize: bool = False):
    """Keep only the links of ``W`` within a regime (block-diagonal ``W``).

    Links between units of different regimes are set to zero; with
    ``row_standardize`` each row is then rescaled to sum to one (rows with
    no remaining link stay zero).

    Examples
    --------
    >>> regime_weights([[0, 1, 1], [1, 0, 1], [1, 1, 0]], [1, 1, 2])
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    """
    W = _mat(W)
    n = _square(W)
    g = list(regimes)
    if len(g) != n:
        raise ValueError("regimes must have length n")
    R = [[W[i][j] if g[i] == g[j] else 0.0 for j in range(n)] for i in range(n)]
    if row_standardize:
        R = [[v / sum(r) if sum(r) else 0.0 for v in r] for r in R]
    return R


def polygon_contiguity(polygons, queen: bool = True, snap: float = 1.4901161193847656e-08):
    """Contiguity neighbours of polygons given as vertex lists.

    As ``spdep::poly2nb``: two polygons are queen neighbours when at least
    one boundary vertex of one lies within ``snap`` (in both coordinates)
    of a boundary vertex of the other, and rook neighbours when at least
    two such vertex pairs exist.  Polygons are lists of ``(x, y)``; a
    repeated closing vertex is ignored.

    :param polygons: Sequence of polygons.
    :param queen: Queen (one shared point) or rook (two).
    :param snap: Coordinate tolerance (default ``sqrt(machine eps)``).
    :return: (n, n) binary list of lists.

    Examples
    --------
    >>> sq = lambda x, y: [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
    >>> polygon_contiguity([sq(0, 0), sq(1, 0), sq(1, 1)], queen=False)
    [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    """
    P = []
    for poly in polygons:
        pts = [(float(a), float(b)) for a, b in poly]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        P.append(pts)
    n = len(P)
    need = 1 if queen else 2
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            hits = 0
            for a in P[i]:
                for b in P[j]:
                    if abs(a[0] - b[0]) <= snap and abs(a[1] - b[1]) <= snap:
                        hits += 1
                        break
                if hits >= need:
                    break
            if hits >= need:
                out[i][j] = out[j][i] = 1.0
    return out


def neighbour_cardinality(W) -> RichResult:
    """Neighbour counts per unit and their frequency table (``spdep::card``).

    Examples
    --------
    >>> r = neighbour_cardinality([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> r.cardinality, r.table
    ([1, 2, 1], {1: 2, 2: 1})
    """
    W = _mat(W)
    n = _square(W)
    card = [sum(1 for j in range(n) if j != i and W[i][j] != 0.0) for i in range(n)]
    table = {}
    for c in sorted(card):
        table[c] = table.get(c, 0) + 1
    return RichResult(
        payload={
            "cardinality": card,
            "table": table,
            "n_links": sum(card),
            "islands": [i for i in range(n) if card[i] == 0],
            "mean": sum(card) / n if n else float("nan"),
        }
    )


def compare_neighbours(W1, W2) -> RichResult:
    """Neighbours present in one weights matrix but not the other (``spdep::diffnb``).

    Examples
    --------
    >>> r = compare_neighbours([[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> r.difference
    [[2], [], [0]]
    """
    A, B = _mat(W1), _mat(W2)
    n = _square(A)
    if _square(B) != n:
        raise ValueError("W1 and W2 must have the same size")
    diff = [[j for j in range(n) if j != i and (A[i][j] != 0.0) != (B[i][j] != 0.0)] for i in range(n)]
    only1 = sum(1 for i in range(n) for j in range(n) if i != j and A[i][j] != 0.0 and B[i][j] == 0.0)
    only2 = sum(1 for i in range(n) for j in range(n) if i != j and B[i][j] != 0.0 and A[i][j] == 0.0)
    return RichResult(
        payload={"difference": diff, "only_first": only1, "only_second": only2, "identical": only1 == 0 and only2 == 0}
    )


def cheatsheet() -> str:
    return "rho_bounds, lag_operator, error_operator, block/regime weights, polygon_contiguity, cardinality, compare_neighbours."
