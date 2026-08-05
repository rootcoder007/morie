# morie.fn -- function file (rootcoder007/morie)
"""Sinkhorn-Knopp doubly stochastic scaling."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_doubly_stoch_proj"]


def ot_doubly_stoch_proj(K, max_iter=200):
    """Rescale a positive matrix to doubly stochastic form.

    Sinkhorn and Knopp proved that any entrywise positive square matrix
    has a unique doubly stochastic scaling ``D1 K D2`` with positive
    diagonal factors, and that alternately normalising rows and columns
    finds it.  The diagonal factors are returned as well because they are
    what the transport modules actually consume.

    Formula: ``K <- diag(1/(K 1)) K``, ``K <- K diag(1/(K' 1))``,
    repeated -- Sinkhorn & Knopp (1967), Theorem 1.

    Parameters
    ----------
    K : array-like, shape (n, n)
        Entrywise positive square matrix.
    max_iter : int, default 200
        Number of row/column sweeps.

    Returns
    -------
    RichResult
        ``M`` (the doubly stochastic scaling), ``iters``, ``d1``, ``d2``,
        ``row_err``, ``col_err``, ``n``.

    References
    ----------
    Sinkhorn, R. and Knopp, P. (1967).  Concerning nonnegative matrices
    and doubly stochastic matrices.  Pacific Journal of Mathematics
    21(2):343-348.  doi:10.2140/pjm.1967.21.343.
    """
    Km = core.mat(K)
    n = len(Km)
    if len(Km[0]) != n:
        raise ValueError("Sinkhorn-Knopp scaling needs a square matrix")
    if any(Km[i][j] <= 0.0 for i in range(n) for j in range(n)):
        raise ValueError("the matrix must be entrywise positive")
    d1 = [1.0] * n
    d2 = [1.0] * n
    it = int(max_iter)
    for _ in range(it):
        for i in range(n):
            s = sum(Km[i][j] * d2[j] for j in range(n))
            d1[i] = 1.0 / s
        for j in range(n):
            s = sum(d1[i] * Km[i][j] for i in range(n))
            d2[j] = 1.0 / s
    M = [[d1[i] * Km[i][j] * d2[j] for j in range(n)] for i in range(n)]
    row_err = max(abs(sum(M[i]) - 1.0) for i in range(n))
    col_err = max(abs(sum(M[i][j] for i in range(n)) - 1.0) for j in range(n))
    return RichResult(payload={
        "M": M, "iters": it, "d1": d1, "d2": d2,
        "row_err": row_err, "col_err": col_err, "n": n,
        "method": "Sinkhorn-Knopp doubly stochastic scaling"})


def cheatsheet():
    return "otdwd: Sinkhorn-Knopp doubly stochastic scaling of a positive matrix"
