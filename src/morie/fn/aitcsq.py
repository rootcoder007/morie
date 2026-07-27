# morie.fn -- function file (rootcoder007/morie)
"""Chi-square statistic and inertia of a table, via correspondence analysis."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["compositional_chisq"]


def compositional_chisq(X, cdf=None, n=None):
    r"""Chi-square and total inertia of a two-way table.

    Follows the correspondence-analysis construction of Nenadic &
    Greenacre (2007). Divide the :math:`I \times J` table :math:`N` by
    its grand total to get the correspondence matrix :math:`P = N/n`;
    let :math:`r` and :math:`c` be its row and column marginals, the row
    and column masses, and :math:`D_r`, :math:`D_c` their diagonal
    matrices. The standardised residuals are

    .. math::

        S = D_r^{-1/2}\,(P - rc')\,D_c^{-1/2}

    Total inertia is :math:`\|S\|_F^2`, and the Pearson statistic is the
    grand total times that inertia:

    .. math::

        \chi^2 = n \sum_{ij} \frac{(p_{ij} - r_i c_j)^2}{r_i c_j}
               = n \,\|S\|_F^2

    which is the usual :math:`\sum (o-e)^2/e` written on the closed
    table. The singular values of :math:`S` come back too: their squares
    are the principal inertias that correspondence analysis decomposes
    the total into.

    A caution on compositional input. Inertia depends only on the row
    profiles, so rescaling rows leaves it unchanged. The chi-square does
    not: it scales with the grand total. Passing rows already closed to
    unit sum makes the grand total the row count, and the statistic then
    describes a table of :math:`I` "observations" rather than the counts
    actually collected. Give ``n`` explicitly in that case, or pass the
    raw counts.

    Parameters
    ----------
    X : array-like, shape (I, J)
        Two-way table of non-negative values, I >= 2 and J >= 2. Counts,
        or any non-negative ratio-scale quantity.
    cdf : callable, optional
        Null CDF of the statistic, replacing the asymptotic chi-square.
    n : float, optional
        Grand total to scale the inertia by. Defaults to ``X.sum()``.
        Supply it when the rows have been closed and the original sample
        size is known.

    Returns
    -------
    RichResult
        keys: ``statistic`` (chi-square), ``p_value``, ``df``,
        ``inertia`` (total), ``principal_inertias``, ``singular_values``,
        ``row_masses``, ``col_masses``, ``n``, ``shape``, ``method``.

    References
    ----------
    Nenadic, O. & Greenacre, M. (2007). Correspondence analysis in R,
    with two- and three-dimensional graphics: the ca package. *Journal
    of Statistical Software*, 20(3), 1-13.

    Greenacre, M. (1984). *Theory and Applications of Correspondence
    Analysis*. Academic Press, London.
    """
    N = np.atleast_2d(np.asarray(X, dtype=float))
    if N.ndim != 2:
        raise ValueError(f"X must be a two-way table; got shape {N.shape}.")
    I, J = N.shape
    if I < 2 or J < 2:
        raise ValueError(f"Need at least a 2x2 table, got {I}x{J}.")
    if not np.all(np.isfinite(N)):
        raise ValueError("X must be finite.")
    if np.any(N < 0):
        raise ValueError("X must be non-negative; a correspondence table has no negative cells.")

    total = float(N.sum())
    if total <= 0:
        raise ValueError("X sums to zero; the correspondence matrix is undefined.")

    P = N / total
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    if np.any(r <= 0) or np.any(c <= 0):
        raise ValueError("Every row and column must have positive mass; drop all-zero rows or columns first.")

    E = np.outer(r, c)
    S = (P - E) / np.sqrt(E)
    inertia = float((S**2).sum())

    grand = total if n is None else float(n)
    if grand <= 0:
        raise ValueError(f"n must be positive, got {grand}.")
    statistic = grand * inertia

    df = (I - 1) * (J - 1)
    p = float(1.0 - cdf(statistic)) if cdf is not None else float(stats.chi2.sf(statistic, df))
    sv = np.linalg.svd(S, compute_uv=False)

    return RichResult(
        title="Chi-square and inertia of a two-way table",
        payload={
            "statistic": statistic,
            "p_value": p,
            "df": int(df),
            "inertia": inertia,
            "principal_inertias": sv**2,
            "singular_values": sv,
            "row_masses": r,
            "col_masses": c,
            "n": grand,
            "shape": (int(I), int(J)),
            "method": "Pearson chi-square via correspondence analysis (Nenadic & Greenacre 2007)",
        },
    )


def cheatsheet():
    return "aitcsq: chi-square and total inertia of a table via correspondence analysis"
