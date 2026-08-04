# morie.fn -- function file (rootcoder007/morie)
"""Spatial factor Poisson log-risk."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["facrisk", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_11_equation_1"]


def facrisk(alpha0, W, phi):
    """Spatial factor Poisson log-risk.

    log(theta_i) = alpha_0 + sum_l w_il phi_l   (Lawson eq. 11.1).

    Poisson risk built from L unobserved components phi with area
    weights W; y_i ~ Pois(e_i theta_i).

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Spatial factor Poisson log-risk", payload=_c.facrisk(alpha0=alpha0, W=W, phi=phi))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_11_equation_1 = facrisk


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp11e1: Spatial factor Poisson log-risk"
