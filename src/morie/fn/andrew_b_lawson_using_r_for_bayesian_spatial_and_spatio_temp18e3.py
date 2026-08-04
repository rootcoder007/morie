# morie.fn -- function file (rootcoder007/morie)
"""Epidemic log-autoregression."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["epiar", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_18_equation_3"]


def epiar(beta0, beta1, i_lag, b1):
    """Epidemic log-autoregression.

    log(f) = b0 + b1 log(I_{i,j-1}) + b1i   (Lawson eq. 18.3).

    Epidemic transmission term: log-linear in the previous period's own
    infective count, with a spatially referenced random effect b1i
    (ICAR prior in the book).

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Epidemic log-autoregression", payload=_c.epiar(beta0=beta0, beta1=beta1, i_lag=i_lag, b1=b1))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_18_equation_3 = epiar


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e3: Epidemic log-autoregression"
