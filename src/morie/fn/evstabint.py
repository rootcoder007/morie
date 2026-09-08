# morie.fn -- function file (rootcoder007/morie)
"""Profile-likelihood CI for GEV/GPD shape ξ.

Implements sec. 2.6.5 applied per sec. 3.3.4 / 4.3.3 of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer (equation checked against the
library PDF).
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_xi_ci_profile"]


def evt_xi_ci_profile(x, alpha=0.05, model="gev"):
    """Profile-likelihood interval for the shape xi:
    {xi : 2[l_p(xi_hat) - l_p(xi)] <= chi2_{1,1-alpha}} (Coles 2001
    sec. 2.6.5). The profile maximizes the remaining parameters at
    each fixed xi by Nelder-Mead."""
    import math
    from . import _sci_core as sci
    from ._stats_core import chi2 as _chi2
    xs = _ev._flat(x)
    crit = float(_chi2.ppf(1.0 - alpha, 1)) / 2.0

    if model == "gev":
        fit = _ev.gev_mle(xs)
        xi_hat, l_hat = fit["xi"], fit["loglik"]

        def prof(xi):
            def nll(th):
                return -_ev.gev_loglik(xs, th[0], math.exp(th[1]), xi)
            r = sci.minimize(nll, [fit["mu"],
                                   math.log(fit["sigma"])],
                             method="Nelder-Mead",
                             options={"maxiter": 2000})
            return -float(r.fun)
    else:
        fit = _ev.gpd_mle(xs)
        xi_hat, l_hat = fit["xi"], fit["loglik"]

        def prof(xi):
            def nll(th):
                return -_ev.gpd_loglik(xs, math.exp(th[0]), xi)
            r = sci.minimize(nll, [math.log(fit["sigma"])],
                             method="Nelder-Mead",
                             options={"maxiter": 2000})
            return -float(r.fun)

    def edge(direction):
        step = 0.01 * direction
        xi = xi_hat
        for _ in range(400):
            xi += step
            if l_hat - prof(xi) > crit:
                # bisect the crossing
                lo, hi = xi - step, xi
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if l_hat - prof(mid) > crit:
                        hi = mid
                    else:
                        lo = mid
                return 0.5 * (lo + hi)
        return xi

    lo = edge(-1.0)
    hi = edge(+1.0)
    res = RichResult(payload={"ci_lo": float(min(lo, hi)),
                              "ci_hi": float(max(lo, hi)),
                              "xi_hat": float(xi_hat),
                              "alpha": float(alpha), "model": model,
                              "method": "profile-likelihood xi interval (Coles 2001 sec. 2.6.5)"})
    return with_describe_pointer(res, "evstabint")


def cheatsheet():
    return "evstabint: Profile-likelihood CI for GEV/GPD shape ξ"


# compact alias per ledger/NAMING.md
evtxiciprofile = evt_xi_ci_profile
