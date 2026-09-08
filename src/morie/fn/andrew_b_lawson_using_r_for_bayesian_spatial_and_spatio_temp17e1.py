# morie.fn -- function file (rootcoder007/morie)
"""Binary spatial regression with random effect."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["logitre", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_17_equation_1"]


def logitre(gamma0, gamma1, d, gamma2, x, R):
    """Binary spatial regression with random effect.

    logit(p_i) = g0 + g1 d_i + g2 x_i + R_i   (Lawson eq. 17.1).

    Binary spatial regression with an exposure d, a covariate x and a
    spatially referenced random effect R.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Binary spatial regression with random effect", payload=_c.logitre(gamma0=gamma0, gamma1=gamma1, d=d, gamma2=gamma2, x=x, R=R))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_17_equation_1 = logitre


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp17e1: Binary spatial regression with random effect"
