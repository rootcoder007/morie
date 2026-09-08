# morie.fn -- function file (rootcoder007/morie)
"""Modulated point-process intensity."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["intmod", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_3"]


def intmod(lam0, lam1):
    """Modulated point-process intensity.

    lambda(s|psi) = lambda_0(s|psi_0) . lambda_1(s|psi_1)   (Lawson eq. 6.3).

    Modulated point-process intensity: a population-at-risk factor times
    an excess-risk factor.  The factorisation is what lets the at-risk
    nuisance be conditioned out.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Modulated point-process intensity", payload=_c.intmod(lam0=lam0, lam1=lam1))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_3 = intmod


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e3: Modulated point-process intensity"
