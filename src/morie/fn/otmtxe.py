# morie.fn -- function file (rootcoder007/morie)
"""RAS matrix scaling to prescribed margins."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['rasscale', 'ot_matrix_scaling']


def rasscale(K, row_target, col_target, max_iter=200):
    """RAS matrix scaling to prescribed margins.

    Alternating Bregman projection onto the row-sum and column-sum constraint sets. It converges when the margins are consistent, and the iteration count is fixed rather than tolerance-driven so the two language arms take exactly the same path; the residual row and column errors are returned so a caller can see whether the fixed budget was enough instead of assuming it was. Zeros in K are preserved -- the sparsity pattern is a constraint, not a starting point.


    Formula: M = diag(u) K diag(v) with u <- r / (K v) and v <- c / (K' u), alternated

    Parameters
    ----------
    K : array-like, shape (m, n)
        Non-negative kernel matrix.
    row_target : array-like
        Required row sums.
    col_target : array-like
        Required column sums.
    max_iter : int
        Fixed number of alternations.

    Returns
    -------
    RichResult
        ``M``, ``u``, ``v``, ``row_error``, ``col_error``, ``iterations``.

    References
    ----------
    Bregman (1967), The relaxation method of finding the common point
    of convex sets, USSR Computational Mathematics and Mathematical
    Physics 7:200-217.  Not held locally; alternating diagonal scaling
    to fixed margins (RAS, Sinkhorn-Knopp) is the standard published
    form of the method.
    """
    K = C.mat(K)
    r = C.vec(row_target); c = C.vec(col_target)
    m = len(K); n = len(K[0])
    if len(r) != m or len(c) != n:
        raise ValueError("targets must match the shape of K")
    if any(v < 0 for row in K for v in row):
        raise ValueError("K must be non-negative")
    if abs(sum(r) - sum(c)) > 1e-9 * max(1.0, abs(sum(r))):
        raise ValueError("row and column targets must have the same total")
    u = [1.0] * m
    v = [1.0] * n
    for _ in range(int(max_iter)):
        for i in range(m):
            s = sum(K[i][j] * v[j] for j in range(n))
            u[i] = r[i] / s if s > 0 else 0.0
        for j in range(n):
            s = sum(K[i][j] * u[i] for i in range(m))
            v[j] = c[j] / s if s > 0 else 0.0
    M = [[u[i] * K[i][j] * v[j] for j in range(n)] for i in range(m)]
    rerr = max(abs(sum(M[i]) - r[i]) for i in range(m))
    cerr = max(abs(sum(M[i][j] for i in range(m)) - c[j]) for j in range(n))
    return RichResult(payload={
        "M": M, "u": u, "v": v, "row_error": rerr, "col_error": cerr,
        "iterations": int(max_iter), "method": "RAS matrix scaling"})


ot_matrix_scaling = rasscale


def cheatsheet():
    return "otmtxe: RAS matrix scaling to prescribed margins."
