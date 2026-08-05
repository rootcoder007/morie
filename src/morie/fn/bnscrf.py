# morie.fn -- function file (rootcoder007/morie)
"""Confidence interval for a partially identified parameter (alias)."""

from ._richresult import RichResult
from .bndfre import bound_frequentist

__all__ = ["bound_credible_interval"]


def bound_credible_interval(lower, upper, alpha=0.05):
    """Alias of :func:`~morie.fn.bndfre.bound_frequentist`.

    The stub carried two module names for one construction: "frequentist
    bound with valid coverage" and "confidence interval for a partially
    identified parameter" are both the Imbens-Manski (2004) equation (6)
    interval.  This name is kept working and forwards; it is not a second
    implementation.

    Parameters
    ----------
    lower, upper : array-like
        Replicated estimates of the lower and upper bound.
    alpha : float, optional
        Miss probability, default 0.05.

    Returns
    -------
    RichResult
        Same payload as :func:`~morie.fn.bndfre.bound_frequentist`, with
        ``method`` naming this entry point.

    References
    ----------
    Imbens, G. W. & Manski, C. F. (2004).  Confidence intervals for
    partially identified parameters.  Econometrica 72(6), 1845-1857,
    equation (6).  doi:10.1111/j.1468-0262.2004.00555.x.
    """
    r = bound_frequentist(lower, upper, alpha)
    p = dict(r)
    p["method"] = "Confidence interval for partially identified parameter"
    return RichResult(payload=p)


def cheatsheet():
    return "bnscrf: alias of bndfre (Imbens-Manski interval)"
