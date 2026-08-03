# morie.fn -- function file (rootcoder007/morie)
"""Spatial lag (SAR/SLM) model by maximum likelihood.

Matches ``spatialreg::lagsarlm``.  Note the spatial regression
functions live in *spatialreg*, not *spdep*, since spdep was split.
"""

from . import _robust_core as _rc
from ._richresult import RichResult, with_describe_pointer

__all__ = ["spatial_lag_model"]


def spatial_lag_model(y, X, W, add_intercept=True):
    """y = rho W y + X beta + eps, fitted by maximum likelihood.

    Ord's concentrated log-likelihood is maximised over rho alone;
    beta and sigma^2 then follow in closed form.  Because W y is
    correlated with the error, ordinary least squares on this model is
    inconsistent -- use this or
    :func:`morie.fn._robust_core.spatial_2sls`. Keys: estimate."""
    r = _rc.spatial_lag_model(y, X, W, add_intercept=add_intercept)
    res = RichResult(payload={"estimate": r["rho"], "rho": r["rho"],
                              "beta": r["beta"], "sigma2": r["sigma2"],
                              "loglik": r["loglik"],
                              "residuals": r["residuals"],
                              "method": r["method"]})
    return with_describe_pointer(res, "sarmod")


def cheatsheet():
    return "sarmod: Spatial lag (SAR/SLM) model by maximum likelihood"
