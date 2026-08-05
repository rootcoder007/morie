# morie.fn -- function file (rootcoder007/morie)
"""Convergent validity: average variance extracted and composite reliability."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["convergent_validity"]


def convergent_validity(loadings, residuals=None):
    """
    Convergent validity of a reflective construct

    Formula: AVE = sum lambda_i^2 / (sum lambda_i^2 + sum theta_i)

    With standardised loadings the residual variance of an indicator is
    theta_i = 1 - lambda_i^2, which is what is used when ``residuals``
    is not supplied.  Composite reliability follows the same algebra,
    CR = (sum lambda_i)^2 / ((sum lambda_i)^2 + sum theta_i).
    Fornell and Larcker's rule is AVE >= 0.5 with CR >= 0.7.

    Parameters
    ----------
    loadings : array-like
        Factor loadings of the indicators on their construct.
    residuals : array-like or None
        Indicator residual (error) variances.  None uses 1 - lambda^2.

    Returns
    -------
    result : dict
        Keys: estimate (AVE), ave, cr, adequate, n_items.

    References
    ----------
    Fornell & Larcker (1981), J. Marketing Research 18(1):39-50.
    """
    lam = core.vec(loadings)
    p = len(lam)
    if p == 0:
        raise ValueError("empty input: no loadings supplied")
    if residuals is None:
        th = [1.0 - v * v for v in lam]
    else:
        th = core.vec(residuals)
        if len(th) != p:
            raise ValueError("loadings and residuals must have the same length")
    for v in th:
        if v < 0.0:
            raise ValueError("residual variances must be non-negative")
    sl2 = sum(v * v for v in lam)
    sth = sum(th)
    sl = sum(lam)
    if sl2 + sth <= 0.0:
        raise ValueError("degenerate construct: total variance is zero")
    ave = sl2 / (sl2 + sth)
    cr = sl * sl / (sl * sl + sth)
    return RichResult(payload={
        "estimate": ave,
        "ave": ave,
        "cr": cr,
        "adequate": 1 if (ave >= 0.5 and cr >= 0.7) else 0,
        "n_items": p,
        "method": "convergent validity: AVE and composite reliability",
    })


def cheatsheet():
    return "convgs: convergent validity (AVE, composite reliability)"


# compact alias per ledger/NAMING.md
convergentvalidity = convergent_validity
