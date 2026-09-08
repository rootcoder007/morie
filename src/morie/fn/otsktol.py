"""Convergence tolerance check for Sinkhorn iterations."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["ot_sinkhorn_tol"]


def ot_sinkhorn_tol(T, a, b):
    """
    Marginal violation of a coupling, in the sup norm.

    Formula: tol = max(|T 1 - a|_inf, |T' 1 - b|_inf)

    Verified against Peyre & Cuturi (2019) Section 4.2 -- source
    consulted: Sinkhorn's stopping criterion is the deviation of the
    current scaling's marginals from the prescribed ``a`` and ``b``.

    Parameters
    ----------
    T : nested sequence
        Coupling matrix.
    a, b : array-like
        Target marginals; each closed to unit mass internally.

    Returns
    -------
    RichResult
        Keys: estimate, row_error, col_error, nrow, ncol, method.

    References
    ----------
    Peyre, G. & Cuturi, M. (2019). Computational Optimal Transport,
    Sec. 4.2.
    """
    m = _big2.mat(T)
    av = list(_big2.pnorm(np.atleast_1d(np.asarray(a, dtype=float))))
    bv = list(_big2.pnorm(np.atleast_1d(np.asarray(b, dtype=float))))
    nr, nc = len(m), len(m[0])
    if len(av) != nr or len(bv) != nc:
        raise ValueError("marginals do not match the shape of T")
    re = max(abs(sum(m[i]) - float(av[i])) for i in range(nr))
    ce = max(abs(sum(m[i][j] for i in range(nr)) - float(bv[j])) for j in range(nc))
    return RichResult(
        payload={
            "estimate": max(re, ce),
            "row_error": re,
            "col_error": ce,
            "nrow": nr,
            "ncol": nc,
            "method": "Sinkhorn marginal violation -- Peyre & Cuturi (2019) Sec. 4.2",
        }
    )


def cheatsheet():
    return "otsktol: Convergence tolerance check for Sinkhorn iterations"


# compact alias per ledger/NAMING.md
otsinkhorntol = ot_sinkhorn_tol
