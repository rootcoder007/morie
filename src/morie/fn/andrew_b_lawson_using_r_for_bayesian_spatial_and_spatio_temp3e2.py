# morie.fn -- function file (rootcoder007/morie)
"""Log-likelihood of independent observations."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["loglksum", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_3_equation_2"]


def loglksum(dens):
    """Log-likelihood of independent observations.

    l(y|theta) = sum_i log f(y_i|theta)   (Lawson eq. 3.2).

    The log of eq. (3.1); the sum form is what is actually evaluated
    because the product underflows for even moderate n.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Log-likelihood of independent observations", payload=_c.loglksum(dens=dens))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_3_equation_2 = loglksum


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp3e2: Log-likelihood of independent observations"
