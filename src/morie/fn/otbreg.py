# morie.fn -- function file (rootcoder007/morie)
"""Iterative Bregman projections onto the transport polytope."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_bregman_proj"]


def ot_bregman_proj(K, a, b, max_iter=200):
    """Alternate KL projections of a Gibbs kernel onto the two marginals.

    Entropic transport is a Bregman projection: the regularised plan is
    the KL-projection of ``K = exp(-C/eps)`` onto the transport polytope
    (eq. 4.7), and the polytope is the intersection of two affine sets,
    so alternating projections converge to it.  Projecting on the rows
    means rescaling each row to sum to ``a``; on the columns, to ``b``.
    That is Sinkhorn's algorithm read as a projection method.

    Formula: ``T <- diag(a / (T 1)) T`` then ``T <- T diag(b / (T' 1))``,
    repeated -- Benamou et al. (2015) eq. (5)-(6); Peyre & Cuturi (2019)
    eq. (4.6)-(4.7).

    Parameters
    ----------
    K : array-like, shape (n, m)
        Gibbs kernel, entrywise positive.
    a : array-like, shape (n,)
        Row marginal.
    b : array-like, shape (m,)
        Column marginal.
    max_iter : int, default 200
        Number of row/column sweeps.  Fixed, not tolerance-driven.

    Returns
    -------
    RichResult
        ``T``, ``iters``, ``row_err``, ``col_err``, ``n``, ``m``.

    References
    ----------
    Benamou, J.-D., Carlier, G., Cuturi, M., Nenna, L. and Peyre, G.
    (2015).  Iterative Bregman projections for regularized transportation
    problems.  SIAM Journal on Scientific Computing 37(2):A1111-A1138.
    doi:10.1137/141000439.
    """
    Km = core.mat(K)
    aa = ot.hist(a)
    bb = ot.hist(b)
    n, m = len(Km), len(Km[0])
    if len(aa) != n or len(bb) != m:
        raise ValueError("kernel does not match the marginals")
    if any(Km[i][j] <= 0.0 for i in range(n) for j in range(m)):
        raise ValueError("the Gibbs kernel must be entrywise positive")
    T = [[Km[i][j] for j in range(m)] for i in range(n)]
    it = int(max_iter)
    for _ in range(it):
        for i in range(n):
            s = sum(T[i])
            f = aa[i] / s if s > 0.0 else 0.0
            for j in range(m):
                T[i][j] *= f
        for j in range(m):
            s = sum(T[i][j] for i in range(n))
            f = bb[j] / s if s > 0.0 else 0.0
            for i in range(n):
                T[i][j] *= f
    row_err = max(abs(sum(T[i]) - aa[i]) for i in range(n))
    col_err = max(abs(sum(T[i][j] for i in range(n)) - bb[j]) for j in range(m))
    return RichResult(payload={
        "T": T, "iters": it, "row_err": row_err, "col_err": col_err,
        "n": n, "m": m,
        "method": "Iterative Bregman projections"})


def cheatsheet():
    return "otbreg: alternating KL projections onto the transport polytope"


# compact alias per ledger/NAMING.md
otbregmanproj = ot_bregman_proj
