# morie.fn -- function file (rootcoder007/morie)
"""Linear marginal structural model.

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

__all__ = ["msm_linear"]


def msm_linear(y, treatment_history, covariate_history=None, weights=None):
    """E[Y(a-bar)] = beta_0 + beta_a a-bar, fitted by weighted least
    squares with the stabilized IPT weights.  beta_a is the average
    causal effect of one additional unit of cumulative treatment;
    time-varying confounders are handled by the weights, not by
    adjustment, which is the whole point of an MSM. Keys: estimate."""
    d = _gp.msm_design(treatment_history)
    f = _gp.msm_weighted_glm(y, d["X"], weights=weights,
                             family="gaussian")
    res = RichResult(payload={"estimate": f["beta"][1],
                              "beta": f["beta"],
                              "beta_a": f["beta"][1],
                              "a_bar": d["a_bar"],
                              "fitted": f["fitted"],
                              "method": "linear MSM (Robins et al. 2000)"})
    return with_describe_pointer(res, "msmlin")


def cheatsheet():
    return "msmlin: Linear marginal structural model"
