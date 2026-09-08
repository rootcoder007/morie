# morie.fn -- function file (rootcoder007/morie)
"""Multilevel Poisson log-rate."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["mlpois", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_15_equation_2"]


def mlpois(beta0, beta1, age, race_effect, v, W):
    """Multilevel Poisson log-rate.

    log(lambda_i) = b0 + b1 age_i + beta(race_i) + v_i + W_i   (Lawson eq. 15.2).

    Multilevel Poisson log-rate: a fixed age slope, a categorical race
    effect already resolved to a per-observation value, an unstructured
    effect v and a spatially structured effect W.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Multilevel Poisson log-rate", payload=_c.mlpois(beta0=beta0, beta1=beta1, age=age, race_effect=race_effect, v=v, W=W))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_15_equation_2 = mlpois


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp15e2: Multilevel Poisson log-rate"
