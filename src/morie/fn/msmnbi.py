# morie.fn -- function file (rootcoder007/morie)
"""Negative-binomial marginal structural model.

Source: Robins, J. M., Hernan, M. A., & Brumback, B. (2000). Marginal
structural models and causal inference in epidemiology. *Epidemiology*
11(5), 550-560; Hernan, M. A. & Robins, J. M. (2020). *Causal
Inference: What If*. CRC.

The stabilized inverse-probability weights are computed by
:func:`morie.fn.msmwt.msmwt`; this module is the weighted outcome
model fitted in the pseudo-population those weights create.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["msm_negative_binomial"]


def msm_negative_binomial(y, treatment_history, covariate_history=None, alpha=1.0,
         offset=None, weights=None):
    """log E[Y(a-bar)] = beta_0 + beta_a a-bar with a
    negative-binomial variance V(mu) = mu + alpha mu^2, which allows
    the overdispersion a Poisson MSM cannot represent (Hilbe 2011).
    The mean model is the same, so the point estimate is obtained by
    the Poisson IRLS and alpha rescales the variance.
    Keys: estimate."""
    d = _gp.msm_design(treatment_history)
    f = _gp.msm_weighted_glm(y, d["X"], weights=weights,
                             family="poisson", offset=offset)
    mu = f["fitted"]
    var = [m + float(alpha) * m * m for m in mu]
    res = RichResult(payload={"estimate": f["beta"][1],
                              "beta": f["beta"],
                              "rate_ratio": math.exp(f["beta"][1]),
                              "variance": var, "alpha": float(alpha),
                              "method": "negative-binomial MSM (Hilbe 2011; Robins et al. 2000)"})
    return with_describe_pointer(res, "msmnbi")


def cheatsheet():
    return "msmnbi: Negative-binomial marginal structural model"
