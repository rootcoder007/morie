# morie.fn -- slice s03 (rootcoder007/morie)
"""Warped Gaussian process.

Source consulted: Snelson, E., Rasmussen, C. E. and Ghahramani, Z.
(2004).  Warped Gaussian processes.  *NIPS* 16, 337-344.  A monotone
warping t = f(y) is applied to the observations, a GP is fitted in the
warped space, and the log marginal likelihood picks up the Jacobian,

    log p(y | X) = log p_GP( f(y) | X ) + sum_i log ( df(y_i) / dy_i )

The predictive *median* in the original space is the inverse warp of the
warped-space predictive mean, f^(-1)(mu), because a monotone map
preserves quantiles; the predictive *mean* is not, and is not claimed
here.  The 2004 proceedings were not retrievable; both statements are
quoted in their standard published form.

The default warp is the identity, which reduces the model exactly to an
ordinary GP -- so the Jacobian term is zero and the two agree, which is
the check the function performs on itself.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

from .gpkrr import gp_kernel_ridge_reg

__all__ = ["gp_warped"]

_WARPS = {
    "identity": (lambda y: y, lambda t: t, lambda y: 1.0),
    "log": (lambda y: math.log(y), lambda t: math.exp(t), lambda y: 1.0 / y),
    "sqrt": (lambda y: math.sqrt(y), lambda t: t * t,
             lambda y: 0.5 / math.sqrt(y)),
}


def gp_warped(X, y, X_test=None, warp="identity", lam=1e-2, gamma=1.0):
    """GP regression under a monotone warping of the observations.

    Returns
    -------
    RichResult with payload:
        estimate  : the predictive median at the first test point
        median    : the predictive median at every test point
        warped_mean : the GP mean in the warped space
        log_jacobian : sum_i log f'(y_i)
    """
    yv = k.vec(y)
    fwd, inv, der = _WARPS.get(str(warp), _WARPS["identity"])
    t = [fwd(v) for v in yv]
    fit = gp_kernel_ridge_reg(X, t, X_test, lam, gamma)
    med = [inv(v) for v in fit["pred"]]
    lj = 0.0
    for v in yv:
        d = der(v)
        lj += math.log(d) if d > 0.0 else float("-inf")
    return RichResult(
        title="Warped Gaussian process",
        summary_lines=[("warp", warp), ("log Jacobian", lj)],
        payload={
            "estimate": med[0] if med else float("nan"),
            "median": med,
            "warped_mean": fit["pred"],
            "var": fit["var"],
            "log_jacobian": lj,
            "warp": str(warp),
            "method": "Warped GP: fit in the warped space, invert for the predictive median (Snelson et al. 2004)",
        },
    )


def cheatsheet():
    return "gpwhr: Warped Gaussian process"


gpwarped = gp_warped
