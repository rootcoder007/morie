# morie.fn -- function file (rootcoder007/morie)
"""Time-varying exposure MSM with stabilized weights.

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

__all__ = ["msm_time_varying_exposure"]


def msm_time_varying_exposure(y, exposure_history, covariate_history=None, time=None,
         weights=None):
    """The stabilized weight
    sw_i = prod_t f(A_t | A-bar_{t-1}) / f(A_t | H_t) reweights the
    sample into a pseudo-population where exposure is independent of
    the measured time-varying confounders, and the MSM
    E[Y(a-bar)] = beta_0 + beta_a a-bar is then fitted by weighted
    least squares (Robins, Hernan and Brumback 2000).  When
    ``weights`` is omitted they are computed from the exposure and
    covariate histories. Keys: estimate."""
    if weights is None and covariate_history is not None:
        from .msmwt import msmwt as _msmwt
        import morie.fn._array_core as _np
        weights = _msmwt(_np.marr(_gp._mat(exposure_history)),
                         _np.marr(_gp._mat(covariate_history)))["sw"]
    d = _gp.msm_design(exposure_history)
    f = _gp.msm_weighted_glm(y, d["X"], weights=weights,
                             family="gaussian")
    ws = f["weights"]
    res = RichResult(payload={"estimate": f["beta"][1],
                              "beta": f["beta"],
                              "a_bar": d["a_bar"],
                              "weight_mean": sum(ws) / len(ws),
                              "weight_max": max(ws),
                              "method": "time-varying exposure MSM (Robins et al. 2000)"})
    return with_describe_pointer(res, "msmtve")


def cheatsheet():
    return "msmtve: Time-varying exposure MSM with stabilized weights"
