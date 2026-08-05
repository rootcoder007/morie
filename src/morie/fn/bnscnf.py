# morie.fn -- function file (rootcoder007/morie)
"""Confidence set for a partially identified parameter (alias)."""

from ._richresult import RichResult
from .bndinf import bound_inference

__all__ = ["bound_confidence_set"]


def bound_confidence_set(theta_grid, moments, alpha=0.05):
    """Alias of :func:`~morie.fn.bndinf.bound_inference`.

    "Confidence set for partial ID" and "inference for partially
    identified parameters" name the same test inversion: the set of
    parameter values a moment-inequality test fails to reject.  This name
    forwards rather than repeating the construction.

    Parameters
    ----------
    theta_grid : array-like
        Candidate parameter values to test.
    moments : array-like, shape (n, 2)
        Interval data, lower end in column 0 and upper end in column 1.
    alpha : float, optional
        Miss probability, default 0.05.

    Returns
    -------
    RichResult
        Same payload as :func:`~morie.fn.bndinf.bound_inference`.

    References
    ----------
    Imbens, G. W. & Manski, C. F. (2004).  Confidence intervals for
    partially identified parameters.  Econometrica 72(6), 1845-1857 --
    the stub's attribution; the set reported is the criterion level set
    of Chernozhukov, Hong & Tamer (2007) as given in equation (4.10) of
    Molinari, F. (2021), Handbook of Econometrics 7A (arXiv:2004.11751
    p. 97).
    """
    r = bound_inference(theta_grid, moments, alpha)
    p = dict(r)
    p["method"] = "Confidence set for partial ID"
    return RichResult(payload=p)


def cheatsheet():
    return "bnscnf: alias of bndinf (test-inversion confidence set)"
