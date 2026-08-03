# morie.fn -- function file (rootcoder007/morie)
"""Logistic marginal structural model.

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

__all__ = ["msm_logistic"]


def msm_logistic(y, treatment_history, covariate_history=None, weights=None):
    """logit P(Y(a-bar) = 1) = beta_0 + beta_a a-bar, fitted by
    weighted IRLS.  exp(beta_a) is a causal odds ratio per unit of
    cumulative treatment. Keys: estimate."""
    d = _gp.msm_design(treatment_history)
    f = _gp.msm_weighted_glm(y, d["X"], weights=weights,
                             family="binomial")
    res = RichResult(payload={"estimate": f["beta"][1],
                              "beta": f["beta"],
                              "odds_ratio": math.exp(f["beta"][1]),
                              "fitted": f["fitted"],
                              "method": "logistic MSM (Robins et al. 2000)"})
    return with_describe_pointer(res, "msmlog")


def cheatsheet():
    return "msmlog: Logistic marginal structural model"
