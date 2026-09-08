# morie.fn -- function file (rootcoder007/morie)
"""Epidemic log-autoregression with neighbours."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["epiarnb", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_18_equation_4"]


def epiarnb(beta0, beta1, i_lag, nb_lag, b1):
    """Epidemic log-autoregression with neighbours.

    log(f) = b0 + b1 log(I_{i,j-1} + sum_{l in delta_i} I_{l,j-1}) + b1i   (eq. 18.4).

    Eq. (18.3) with the neighbourhood term added: ``nb_lag[i]`` is the
    already-summed lagged infective count of the regions adjacent to i.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Epidemic log-autoregression with neighbours", payload=_c.epiarnb(beta0=beta0, beta1=beta1, i_lag=i_lag, nb_lag=nb_lag, b1=b1))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_18_equation_4 = epiarnb


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e4: Epidemic log-autoregression with neighbours"
