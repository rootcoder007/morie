# morie.fn -- function file (rootcoder007/morie)
"""ER subscale composite reliability (rho_c)."""

from __future__ import annotations

from morie.fn._containers import ESRes

from . import _array_core as np
from . import _frame_core as pd


def subscale_er_composite_rel(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> ESRes:
    """Composite reliability (rho_c) for the ER subscale.

    rho_c = (sum lambda_j)^2 / ((sum lambda_j)^2 + sum(1 - lambda_j^2)), where lambda_j are the standardised loadings of a
    one-factor model fitted to the item correlation matrix by maximum
    likelihood (Rubin-Thayer EM, the solution ``stats::factanal`` reports).
    Loadings taken from the first principal component -- what this function
    used to use -- are not factor loadings: their squares average to
    lambda_1 / p, the share of variance on the first component, which
    overstates both AVE and rho_c.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item responses; a DataFrame keeps its complete rows of ``items``.
    items : list[str], optional
        Column names. Default: ER1-ER5.

    Returns
    -------
    ESRes
        measure="composite_reliability_ER"; ``extra``: ``loadings`` (absolute standardised),
        ``uniquenesses``, ``subscale``, ``iterations``, ``converged``.

    References
    ----------
    Fornell, C. and Larcker, D. F. (1981). Evaluating structural equation
    models with unobservable variables and measurement error. Journal of
    Marketing Research 18, 39-50.
    Rubin, D. B. and Thayer, D. T. (1982). EM algorithms for ML factor
    analysis. Psychometrika 47, 69-76.

    Examples
    --------
    >>> rows = [[2, 3, 1, 4], [3, 3, 2, 5], [4, 5, 3, 4], [1, 2, 2, 2], [5, 4, 4, 5], [2, 1, 3, 3], [3, 4, 2, 3], [4, 3, 5, 4]]
    >>> round(subscale_er_composite_rel(rows).estimate, 10)
    0.8651078397
    """
    from ._onefactor import ave_and_cr, subscale_matrix

    rows = subscale_matrix(data, items, "ER")
    fit = ave_and_cr(rows)
    return ESRes(
        measure="composite_reliability_ER",
        estimate=float(fit["cr"]),
        n=len(rows),
        extra={
            "loadings": fit["loadings"],
            "uniquenesses": fit["uniquenesses"],
            "subscale": "ER",
            "iterations": fit["iterations"],
            "converged": fit["converged"],
        },
    )


cr_er = subscale_er_composite_rel


def cheatsheet() -> str:
    return "subscale_er_composite_rel({}) -> ER subscale composite reliability (rho_c)."
