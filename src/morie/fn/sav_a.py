# morie.fn -- function file (rootcoder007/morie)
"""EA subscale average variance extracted."""

from __future__ import annotations

from morie.fn._containers import ESRes

from . import _array_core as np
from . import _frame_core as pd


def subscale_ea_ave(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> ESRes:
    """Average Variance Extracted (AVE) for the EA subscale.

    AVE = mean(lambda_j^2), where lambda_j are the standardised loadings of a
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
        Column names. Default: EA1-EA5.

    Returns
    -------
    ESRes
        measure="AVE_EA"; ``extra``: ``loadings`` (absolute standardised),
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
    >>> round(subscale_ea_ave(rows).estimate, 10)
    0.6222069522
    """
    from ._onefactor import ave_and_cr, subscale_matrix

    rows = subscale_matrix(data, items, "EA")
    fit = ave_and_cr(rows)
    return ESRes(
        measure="AVE_EA",
        estimate=float(fit["ave"]),
        n=len(rows),
        extra={
            "loadings": fit["loadings"],
            "uniquenesses": fit["uniquenesses"],
            "subscale": "EA",
            "iterations": fit["iterations"],
            "converged": fit["converged"],
        },
    )


ave_ea = subscale_ea_ave


def cheatsheet() -> str:
    return "subscale_ea_ave({}) -> EA subscale average variance extracted."


# compact alias per ledger/NAMING.md
subscaleeaave = subscale_ea_ave
