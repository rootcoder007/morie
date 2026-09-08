# morie.fn -- function file (rootcoder007/morie)
"""Semi-parametric tail / extreme regression.

Implements sec. 6.1-6.2 (nonstationary GEV with a linear trend) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer (equation checked against the
library PDF).
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["semiparametric_max"]


def semiparametric_max(x, t=None):
    """Nonstationary GEV with a linear trend in location,
    mu(t) = beta0 + beta1 t (Coles 2001 sec. 6.2, the eq. 6.1-style
    linear-trend model), fitted by maximum likelihood. ``estimate`` is
    the trend coefficient beta1; a likelihood-ratio statistic against
    the stationary model (sec. 6.2.3 practice) is included."""
    import math
    from . import _sci_core as sci
    xs = _ev._flat(x)
    n = len(xs)
    ts = [float(v) for v in (range(n) if t is None else _ev._flat(t))]
    tbar = sum(ts) / n
    tsd = math.sqrt(sum((v - tbar) ** 2 for v in ts) / n) or 1.0
    tz = [(v - tbar) / tsd for v in ts]

    f0 = _ev.gev_mle(xs)

    def nll(th):
        b0, b1, ls, xi = th
        s = math.exp(ls)
        return -sum(_ev.gev_logpdf(xs[i], b0 + b1 * tz[i], s, xi)
                    for i in range(n))

    r = sci.minimize(nll, [f0["mu"], 0.0, math.log(f0["sigma"]),
                           f0["xi"]],
                     method="Nelder-Mead",
                     options={"maxiter": 6000})
    b0, b1, ls, xi = [float(v) for v in r.x]
    ll1 = -float(r.fun)
    lr = 2.0 * (ll1 - f0["loglik"])
    beta1 = b1 / tsd          # back to the original time scale
    res = RichResult(payload={"estimate": beta1, "beta0": b0,
                              "beta1": beta1,
                              "sigma": math.exp(ls), "xi": xi,
                              "ll": ll1, "lr_vs_stationary": lr,
                              "method": "nonstationary GEV, linear trend in mu (Coles 2001 sec. 6.2)"})
    return with_describe_pointer(res, "smt")


def cheatsheet():
    return "smt: Semi-parametric tail / extreme regression"
