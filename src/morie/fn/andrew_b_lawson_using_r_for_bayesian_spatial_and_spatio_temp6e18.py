# morie.fn -- function file (rootcoder007/morie)
"""Log-Gaussian Cox process intensity."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["lgcpint", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_18"]


def lgcpint(lam0, beta, S):
    """Log-Gaussian Cox process intensity.

    lambda(s) = lambda_0(s) exp{beta + S(s)}   (Lawson eq. 6.18).

    First-order intensity of the Diggle et al. (1998) log-Gaussian Cox
    process: a modulating baseline, a non-zero mean level beta, and a
    zero-mean Gaussian process S(s) supplied by the caller.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Log-Gaussian Cox process intensity", payload=_c.lgcpint(lam0=lam0, beta=beta, S=S))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_18 = lgcpint


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e18: Log-Gaussian Cox process intensity"
