# morie.fn -- slice s03 (rootcoder007/morie)
"""Effective number of parameters p_D from a deviance sample.

Source consulted: Spiegelhalter, D. J., Best, N. G., Carlin, B. P. and
van der Linde, A. (2002).  Bayesian measures of model complexity and
fit.  *Journal of the Royal Statistical Society B* 64(4), 583-639,
equation (9):

    p_D = Dbar - D(thetabar)

the posterior mean deviance minus the deviance at the posterior mean.
The paper is paywalled; the equation is quoted in its standard
published form.  The companion measure p_V = var(D)/2 (Gelman, Carlin,
Stern and Rubin, *Bayesian Data Analysis*, 2nd ed., section 6.7) is
returned alongside, since it needs only the deviance sample and is the
fallback when D(thetabar) is unavailable.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["effective_parameters_dic"]


def effective_parameters_dic(deviance, d_at_mean=None):
    """Effective parameter count p_D (and p_V) from posterior deviance draws.

    Parameters
    ----------
    deviance : array-like
        Posterior draws of the deviance.
    d_at_mean : float, optional
        D(thetabar).  When omitted, ``estimate`` is p_V.

    Returns
    -------
    RichResult with payload:
        estimate : p_D if d_at_mean given, else p_V
        p_d, p_v, dbar, d_hat, variant
    """
    d = k.vec(deviance)
    dbar = k.mean(d)
    pv = 0.5 * k.variance(d, 1)
    if d_at_mean is None:
        pd = float("nan")
        dhat = float("nan")
        est = pv
        variant = "p_V"
    else:
        dhat = float(d_at_mean)
        pd = dbar - dhat
        est = pd
        variant = "p_D"
    return RichResult(
        title="Effective number of parameters",
        summary_lines=[("p_D", pd), ("p_V", pv)],
        payload={
            "estimate": est,
            "p_d": pd,
            "p_v": pv,
            "dbar": dbar,
            "d_hat": dhat,
            "variant": variant,
            "n": len(d),
            "method": "Effective parameters from the deviance sample (p_D, p_V)",
        },
    )


def cheatsheet():
    return "pdic: Effective parameters from DIC (p_D)"
