# morie.fn -- function file (rootcoder007/morie)
"""GMM estimator for a marginal structural model.

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

__all__ = ["msm_gmm_estimator"]


def msm_gmm_estimator(y, treatment_history, covariate_history=None,
         instruments=None, weights=None):
    """E[Z (Y - g(a-bar; beta))] = 0 with IPT-weighted moments
    (Hansen 1982; Robins 1999).  The instruments Z default to the
    design itself, which makes the system exactly identified and
    recovers the weighted least squares solution; supplying genuine
    instruments handles unmeasured confounding of the treatment.
    Keys: estimate."""
    d = _gp.msm_design(treatment_history)
    Z = d["X"] if instruments is None else \
        [[1.0] + list(r) for r in _gp._mat(instruments)]
    f = _gp.msm_gmm(y, d["X"], Z, weights=weights)
    res = RichResult(payload={"estimate": f["beta"][1],
                              "beta": f["beta"],
                              "moments": f["moments"],
                              "method": "GMM estimator for an MSM (Hansen 1982; Robins 1999)"})
    return with_describe_pointer(res, "msmgmm")


def cheatsheet():
    return "msmgmm: GMM estimator for a marginal structural model"
