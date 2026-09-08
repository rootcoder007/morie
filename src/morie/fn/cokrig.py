# morie.fn -- function file (rootcoder007/morie)
"""Ordinary co-kriging with a secondary variable.

Solves the co-kriging system under both unbiasedness constraints.
"""

from . import _robust_core as _rc
from ._richresult import RichResult, with_describe_pointer

__all__ = ["cokriging"]


def cokriging(coords, z1, z2, s0, cross_vario=None, model=None):
    """Z1*(s0) = sum lambda_i Z1(s_i) + sum mu_j Z2(s_j).

    The covariate earns its place only through the cross-variogram:
    set it to zero and the mu weights vanish, leaving ordinary kriging
    on Z1.  Constraints are sum lambda = 1 and sum mu = 0, which is
    what keeps the predictor unbiased. Keys: estimate."""
    r = _rc.cokriging(coords, z1, z2, s0, cross_vario=cross_vario,
                      model=model)
    res = RichResult(payload={"estimate": r["prediction"],
                              "prediction": r["prediction"],
                              "variance": r["variance"],
                              "lambda": r["lambda"], "mu": r["mu"],
                              "method": r["method"]})
    return with_describe_pointer(res, "cokrig")


def cheatsheet():
    return "cokrig: Ordinary co-kriging with a secondary variable"
