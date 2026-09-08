# morie.fn -- function file (rootcoder007/morie)
"""Spatial error (SEM) model by maximum likelihood.

Matches ``spatialreg::errorsarlm``.
"""

from . import _robust_core as _rc
from ._richresult import RichResult, with_describe_pointer

__all__ = ["spatial_error_model"]


def spatial_error_model(y, X, W, add_intercept=True):
    """y = X beta + u with u = lambda W u + eps, by maximum likelihood.

    The likelihood is concentrated by spatially filtering both sides,
    y* = y - lambda W y and X* = X - lambda W X.  Spatial dependence
    here is a nuisance in the errors rather than a substantive lag, so
    beta keeps its usual interpretation. Keys: estimate."""
    r = _rc.spatial_error_model(y, X, W, add_intercept=add_intercept)
    res = RichResult(payload={"estimate": r["lambda"],
                              "lambda": r["lambda"], "beta": r["beta"],
                              "sigma2": r["sigma2"],
                              "loglik": r["loglik"],
                              "residuals": r["residuals"],
                              "method": r["method"]})
    return with_describe_pointer(res, "semmod")


def cheatsheet():
    return "semmod: Spatial error (SEM) model by maximum likelihood"
