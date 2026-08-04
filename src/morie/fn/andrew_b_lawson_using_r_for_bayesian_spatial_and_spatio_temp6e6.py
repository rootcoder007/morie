# morie.fn -- function file (rootcoder007/morie)
"""Case-control logistic likelihood."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["cclogl", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_6"]


def cclogl(eta, y):
    """Case-control logistic likelihood.

    L = prod_i {exp(eta_i)}^{y_i} / (1 + exp(eta_i))   (Lawson eq. 6.6).

    Case-control likelihood for a case event model, which reduces to a
    logistic likelihood: the at-risk population function drops out.
    Returned on the log scale, with the case probabilities.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Case-control logistic likelihood", payload=_c.cclogl(eta=eta, y=y))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_6 = cclogl


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e6: Case-control logistic likelihood"
