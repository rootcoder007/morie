"""Spatial weights matrices from point coordinates."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["weights_matrix"]


def _dist(a, b):
    s = 0.0
    for k in range(len(a)):
        d = a[k] - b[k]
        s += d * d
    return np.sqrt(s)


def weights_matrix(coords, method="distance", k_or_threshold=1,
                   alpha=1.0, row_standardize=True):
    """
    Spatial weights matrix construction

    Formula: W = [w_ij], w_ii = 0, optionally row-standardised to
    sum_j w_ij = 1

    The spatial weights matrix formalises "neighbour" for a spatial
    model: ``w_ij`` is non-zero when j is a neighbour of i, and zero
    otherwise.  Everything downstream -- the spatial lag ``Wy``, Moran's
    I, a spatial autoregressive model -- is defined relative to this
    choice, which is why the choice is an assumption rather than an
    estimate.

    Three coordinate-based schemes are offered:

    ``"distance"``
        Distance band.  ``w_ij = 1`` when ``0 < d_ij <= threshold``, else
        0, with ``threshold = k_or_threshold``.  A unit with no
        neighbour inside the band keeps an all-zero row.

    ``"knn"``
        k nearest neighbours.  ``w_ij = 1`` for the ``k`` units nearest
        to i, else 0, with ``k = k_or_threshold``.  This relation is not
        symmetric: j may be among i's k nearest without i being among
        j's.

    ``"inverse"``
        Inverse distance.  ``w_ij = d_ij ** (-alpha)`` for
        ``0 < d_ij <= threshold``, else 0.  ``threshold`` may be
        infinite, which connects every pair.

    Distances are Euclidean.  The diagonal is always zero: a unit is not
    its own neighbour.  With ``row_standardize`` the rows that have at
    least one neighbour are divided by their sum, so that a spatial lag
    is a weighted average of neighbours; all-zero rows are left alone.

    Contiguity weights (rook, queen) are not constructed here.  They need
    a lattice or a polygon topology rather than points -- two units are
    rook-contiguous when their boundaries share an edge, which a
    coordinate pair does not determine.

    Parameters
    ----------
    coords : array-like
        ``(n, d)`` coordinates, one row per unit.  A 1-D input is read as
        ``n`` points on a line.
    method : {"distance", "knn", "inverse"}
        Weighting scheme.
    k_or_threshold : float or int
        The distance band for ``"distance"`` and ``"inverse"``, or the
        number of neighbours ``k`` for ``"knn"``.
    alpha : float
        Distance decay exponent for ``"inverse"``.  Default 1.
    row_standardize : bool
        Divide each non-empty row by its sum.  Default True.

    Returns
    -------
    result : RichResult
        Keys: weights, n, method, scheme, row_standardized, n_links,
        n_islands, pct_nonzero.

    See Also
    --------
    morie.fn.swrook.swrook, morie.fn.swqueen.swqueen : rook and queen
        contiguity weights for a regular lattice.

    References
    ----------
    Anselin L (1988).  Spatial Econometrics: Methods and Models.  Kluwer,
    Dordrecht.  Chapter 3 defines the weights matrix, the row-standardised
    form and the spatial lag built from it.
    """
    arr = np.asarray(coords, dtype=float)
    lst = arr.tolist()
    if len(lst) == 0:
        raise ValueError("coords must be non-empty")
    if not isinstance(lst[0], list):
        pts = [[float(v)] for v in lst]
    else:
        pts = [[float(v) for v in row] for row in lst]
    n = len(pts)
    if n < 2:
        raise ValueError("need at least two units")
    dim = len(pts[0])
    for row in pts:
        if len(row) != dim:
            raise ValueError("all coordinate rows must have the same length")
    if method not in ("distance", "knn", "inverse"):
        raise ValueError("method must be 'distance', 'knn' or 'inverse'; "
                         "rook and queen contiguity need a lattice, see "
                         "morie.fn.swrook and morie.fn.swqueen")

    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = _dist(pts[i], pts[j])
            d[i][j] = v
            d[j][i] = v

    w = [[0.0] * n for _ in range(n)]
    if method == "knn":
        k = int(k_or_threshold)
        if k < 1 or k > n - 1:
            raise ValueError("k must lie between 1 and n - 1")
        for i in range(n):
            order = sorted((j for j in range(n) if j != i),
                           key=lambda j: (d[i][j], j))
            for j in order[:k]:
                w[i][j] = 1.0
    else:
        thr = float(k_or_threshold)
        if not (thr > 0.0):
            raise ValueError("threshold must be positive")
        alpha = float(alpha)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dij = d[i][j]
                if dij <= 0.0 or dij > thr:
                    continue
                w[i][j] = 1.0 if method == "distance" else dij ** (-alpha)

    n_links = 0
    n_islands = 0
    for i in range(n):
        rs = 0.0
        cnt = 0
        for j in range(n):
            if w[i][j] != 0.0:
                rs += w[i][j]
                cnt += 1
        n_links += cnt
        if cnt == 0:
            n_islands += 1
        elif row_standardize:
            for j in range(n):
                if w[i][j] != 0.0:
                    w[i][j] = w[i][j] / rs

    return RichResult(
        payload={
            "weights": w,
            "n": n,
            "scheme": method,
            "row_standardized": bool(row_standardize),
            "n_links": n_links,
            "n_islands": n_islands,
            "pct_nonzero": 100.0 * n_links / float(n * n),
            "method": "Spatial weights matrix (Anselin 1988, ch.3)",
        }
    )


def cheatsheet():
    return "wmtwgt: spatial weights matrix from coordinates (Anselin 1988)"


# compact alias per ledger/NAMING.md
wgtmatrix = weights_matrix
