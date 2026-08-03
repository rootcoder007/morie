# morie.fn -- function file (rootcoder007/morie)
"""Marginal structural Cox model.

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

__all__ = ["msm_cox_marginal"]


def msm_cox_marginal(time, event, treatment_history, covariate_history=None,
         weights=None):
    """lambda(t | a-bar) = lambda_0(t) exp(beta_a a-bar) fitted by a
    weighted partial likelihood (Robins, Hernan and Brumback 2000;
    Hernan, Brumback and Robins 2000).  Weighting the partial
    likelihood by the stabilized IPT weights makes exp(beta_a) a
    marginal (causal) hazard ratio rather than a conditional one.
    Keys: estimate."""
    f = _gp.msm_cox_weighted(time, event, treatment_history,
                             weights=weights)
    res = RichResult(payload={"estimate": f["beta"],
                              "beta": f["beta"],
                              "hazard_ratio": f["hazard_ratio"],
                              "method": "marginal structural Cox model (Robins et al. 2000)"})
    return with_describe_pointer(res, "msmcox")


def cheatsheet():
    return "msmcox: Marginal structural Cox model"
