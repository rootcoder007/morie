# morie.fn -- function file (rootcoder007/morie)
"""Measurement-error normal outcome model."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["menorm", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_16_equation_1"]


def menorm(beta0, beta1, x_true, tau):
    """Measurement-error normal outcome model.

    y_i ~ N(mu_i, tau^-1),  mu_i = b0 + b1 x^T_i   (Lawson eq. 16.1).

    The outcome half of the classical measurement-error model: the
    regression is on the unobserved true covariate x^T, not on the
    error-prone x.  Returns the mean and the precision-implied scale.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Measurement-error normal outcome model", payload=_c.menorm(beta0=beta0, beta1=beta1, x_true=x_true, tau=tau))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_16_equation_1 = menorm


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp16e1: Measurement-error normal outcome model"
