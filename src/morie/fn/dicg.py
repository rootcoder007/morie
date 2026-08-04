# morie.fn -- slice s03 (rootcoder007/morie)
"""Deviance information criterion.

Source consulted: Spiegelhalter, D. J., Best, N. G., Carlin, B. P. and
van der Linde, A. (2002).  Bayesian measures of model complexity and
fit (with discussion).  *Journal of the Royal Statistical Society B*
64(4), 583-639.  Their equations (9) and (10) define

    p_D  = Dbar - D(thetabar)
    DIC  = Dbar + p_D  =  D(thetabar) + 2 p_D

with Dbar the posterior mean of the deviance and D(thetabar) the
deviance evaluated at the posterior mean of the parameters.  The 2002
JRSS-B paper is paywalled, so the two equations are quoted in their
standard published form; both are reproduced identically wherever DIC
is defined.

This function takes the posterior sample of the deviance directly, so
Dbar is its mean.  D(thetabar) must be supplied, since it cannot be
recovered from the deviance sample alone -- when it is omitted the
alternative complexity measure p_V = var(D)/2 of Gelman et al. is used
and reported as such.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["deviance_information_criterion"]


def deviance_information_criterion(deviance, d_at_mean=None):
    """DIC from a posterior sample of the deviance.

    Parameters
    ----------
    deviance : array-like
        Posterior draws of the deviance D(theta) = -2 log p(y | theta).
    d_at_mean : float, optional
        D(thetabar), the deviance at the posterior mean.  If omitted,
        p_D is replaced by p_V = var(D) / 2.

    Returns
    -------
    RichResult with payload:
        estimate : DIC = Dbar + p_D
        dbar     : posterior mean deviance
        p_d      : effective number of parameters
        d_hat    : D(thetabar), or Dbar - p_V when not supplied
        variant  : "p_D" or "p_V"
    """
    d = k.vec(deviance)
    dbar = k.mean(d)
    if d_at_mean is None:
        pd = 0.5 * k.variance(d, 1)
        dhat = dbar - pd
        variant = "p_V"
    else:
        dhat = float(d_at_mean)
        pd = dbar - dhat
        variant = "p_D"
    dic = dbar + pd
    return RichResult(
        title="Deviance information criterion",
        summary_lines=[("DIC", dic), ("p_D", pd), ("Dbar", dbar)],
        payload={
            "estimate": dic,
            "dbar": dbar,
            "p_d": pd,
            "d_hat": dhat,
            "variant": variant,
            "n": len(d),
            "method": "Spiegelhalter et al (2002) deviance information criterion",
        },
    )


def cheatsheet():
    return "dicg: Deviance information criterion (DIC)"
