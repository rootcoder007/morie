# morie.fn -- function file (rootcoder007/morie)
"""Contextual multilevel logit predictor."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["mlogitlp", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_8"]


def mlogitlp(f, g, R):
    """Contextual multilevel logit predictor.

    logit(p_i) = f_i + g_i + R_i   (Lawson eq. 6.8).

    Contextual multilevel logit: an individual-predictor term, a
    spatial-unit covariate term, and a spatial-unit random effect.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Contextual multilevel logit predictor", payload=_c.mlogitlp(f=f, g=g, R=R))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_8 = mlogitlp


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e8: Contextual multilevel logit predictor"
