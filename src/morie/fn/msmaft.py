# morie.fn -- function file (rootcoder007/morie)
"""Marginal structural accelerated failure time model.

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

__all__ = ["msm_accelerated_failure"]


def msm_accelerated_failure(time, event, treatment_history, covariate_history=None,
         weights=None):
    """log T(a-bar) = beta_0 + beta_a a-bar + eps, the structural
    accelerated failure time model (Robins and Tsiatis 1991).  Fitted
    on the log scale by weighted least squares over the uncensored
    observations; exp(beta_a) is the ratio by which cumulative
    treatment multiplies survival time. Keys: estimate."""
    ts = _gp._flat(time)
    ev = _gp._flat(event)
    d = _gp.msm_design(treatment_history)
    idx = [i for i in range(len(ts)) if ev[i] > 0 and ts[i] > 0]
    if not idx:
        raise ValueError("need at least one uncensored positive time")
    w = None if weights is None else [
        _gp._flat(weights)[i] for i in idx]
    f = _gp.msm_weighted_glm([math.log(ts[i]) for i in idx],
                             [d["X"][i] for i in idx],
                             weights=w, family="gaussian")
    res = RichResult(payload={"estimate": f["beta"][1],
                              "beta": f["beta"],
                              "time_ratio": math.exp(f["beta"][1]),
                              "n_uncensored": len(idx),
                              "method": "structural AFT model (Robins-Tsiatis 1991)"})
    return with_describe_pointer(res, "msmaft")


def cheatsheet():
    return "msmaft: Marginal structural accelerated failure time model"
