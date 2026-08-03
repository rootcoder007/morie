# morie.fn -- function file (rootcoder007/morie)
"""Poisson marginal structural model.

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

__all__ = ["msm_poisson"]


def msm_poisson(y, treatment_history, covariate_history=None, offset=None,
         weights=None):
    """log E[Y(a-bar)] = beta_0 + beta_a a-bar with an optional
    log-exposure offset, fitted by weighted IRLS.  exp(beta_a) is a
    causal rate ratio. Keys: estimate."""
    d = _gp.msm_design(treatment_history)
    f = _gp.msm_weighted_glm(y, d["X"], weights=weights,
                             family="poisson", offset=offset)
    res = RichResult(payload={"estimate": f["beta"][1],
                              "beta": f["beta"],
                              "rate_ratio": math.exp(f["beta"][1]),
                              "fitted": f["fitted"],
                              "method": "Poisson MSM (Robins et al. 2000)"})
    return with_describe_pointer(res, "msmpoi")


def cheatsheet():
    return "msmpoi: Poisson marginal structural model"
