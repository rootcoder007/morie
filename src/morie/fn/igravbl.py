# morie.fn -- function file (rootcoder007/morie)
"""Gravity model row/column balancing (IPFP)."""

from __future__ import annotations

from . import _gravity as gv
from ._containers import SpatialResult


def igravbl(flow_matrix, row_totals, col_totals, tol=1e-12, max_iter=10000):
    r"""Gravity model row/column balancing (IPFP).

    Iterative proportional fitting (Deming and Stephan 1940): alternately
    rescale the rows of a seed flow matrix to the origin totals and its
    columns to the destination totals until the row sums match to ``tol``.
    The limit is the doubly constrained matrix closest to the seed in
    Kullback-Leibler divergence -- the balancing factors ``A_i``, ``B_j`` of
    a doubly constrained gravity model (Wilson 1970) -- and equals
    ``stats::loglin(..., start = seed, fit = TRUE)``.

    Parameters
    ----------
    flow_matrix : array-like, shape (I, J)
        Non-negative seed matrix (e.g. unconstrained gravity predictions).
    row_totals, col_totals : array-like
        Target origin and destination totals (equal sums).
    tol : float
        Relative tolerance on the row sums.
    max_iter : int
        Iteration cap.

    Returns
    -------
    SpatialResult
        ``statistic`` is the largest remaining row-total error; ``extra``
        has the ``balanced`` matrix and ``iterations``.

    References
    ----------
    Deming, W. E. and Stephan, F. F. (1940). On a least squares adjustment of a sampled frequency
    table when the expected marginal totals are known. *Annals of Mathematical Statistics*, 11(4),
    427-444.

    Wilson, A. G. (1970). *Entropy in Urban and Regional Modelling*. Pion, London.

    Examples
    --------
    >>> r = igravbl([[1, 2], [3, 4]], [5, 5], [6, 4])
    >>> [[round(v, 10) for v in row] for row in r.extra["balanced"]]
    [[2.7576506721, 2.2423493279], [3.2423493279, 1.7576506721]]
    """
    T, err, it = gv.ipf(flow_matrix, row_totals, col_totals, tol, max_iter)
    return SpatialResult(name="igravbl", statistic=err, extra={"balanced": T, "iterations": it})


igravbl_fn = igravbl


def cheatsheet() -> str:
    return "igravbl(M, rows, cols) -> IPF-balanced matrix (Deming-Stephan)"
