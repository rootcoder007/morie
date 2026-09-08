# morie.fn -- function file (rootcoder007/morie)
"""Delta-method CI for a return level z_T.

Implements sec. 3.3.3 of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_return_level_ci"]


def evt_return_level_ci(x, T, alpha=0.05):
    """Return level with delta-method confidence interval:
    Var(z_T) = grad^T V grad with V the MLE covariance (Coles 2001
    sec. 3.3.3). Fits the GEV to ``x`` first."""
    import math
    from ._stats_core import norm as _norm
    f = _ev.gev_mle(x)
    z = _ev.gev_return_level(float(T), f["mu"], f["sigma"], f["xi"])
    g = _ev.gev_return_level_grad(float(T), f["mu"], f["sigma"],
                                  f["xi"])
    V = f["cov"]
    var = sum(g[i] * V[i][j] * g[j]
              for i in range(3) for j in range(3))
    se = math.sqrt(max(var, 0.0))
    zc = float(_norm.ppf(1.0 - alpha / 2.0))
    res = RichResult(payload={"z_T": float(z),
                              "ci_lo": float(z - zc * se),
                              "ci_hi": float(z + zc * se),
                              "se": se, "T": float(T),
                              "method": "delta-method return-level CI (Coles 2001 sec. 3.3.3)"})
    return with_describe_pointer(res, "evrlci")


def cheatsheet():
    return "evrlci: Delta-method CI for a return level z_T"
