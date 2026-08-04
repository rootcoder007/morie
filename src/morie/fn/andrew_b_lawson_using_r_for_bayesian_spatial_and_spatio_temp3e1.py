# morie.fn -- function file (rootcoder007/morie)
"""Joint likelihood of independent observations."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["likprod", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_3_equation_1"]


def likprod(dens):
    """Joint likelihood of independent observations.

    L(y|theta) = prod_i f(y_i|theta)   (Lawson eq. 3.1 p.--).

    The joint likelihood of conditionally independent observations is
    the product of the individual contributions.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Joint likelihood of independent observations", payload=_c.likprod(dens=dens))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_3_equation_1 = likprod


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp3e1: Joint likelihood of independent observations"
