# morie.fn -- function file (rootcoder007/morie)
"""Bayesian GPD posterior via Metropolis.

Implements sec. 9.1.2-9.1.3 applied to the threshold model of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer (equation checked against the
library PDF).
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_bayes_gpd"]


def evt_bayes_gpd(y, n_draws=2000, seed=42, prior_sd=(10.0, 1.0)):
    """Random-walk Metropolis on (log sigma, xi) targeting the GPD
    posterior with vague normal priors (Coles 2001 sec. 9.1.3 recipe
    on the ch. 4 threshold likelihood)."""
    import math
    ys = _ev._flat(y)
    f = _ev.gpd_mle(ys)
    rng = np.random.default_rng(seed)
    s_ls, s_xi = prior_sd

    def logpost(ls, xi):
        lp = _ev.gpd_loglik(ys, math.exp(ls), xi)
        lp += -ls * ls / (2 * s_ls * s_ls)
        lp += -xi * xi / (2 * s_xi * s_xi)
        return lp

    th = [math.log(f["sigma"]), f["xi"]]
    step = [0.08, 0.05]
    lp = logpost(*th)
    warm = max(200, int(n_draws) // 2)
    draws = []
    acc = tot = 0
    for it in range(warm + int(n_draws)):
        for j in range(2):
            prop = list(th)
            prop[j] += step[j] * float(rng.normal())
            lp_new = logpost(*prop)
            tot += 1
            if math.log(max(float(rng.uniform(0, 1)), 1e-300))                     < lp_new - lp:
                th, lp = prop, lp_new
                acc += 1
                if it < warm:
                    step[j] *= 1.05
            elif it < warm:
                step[j] *= 0.97
        if it >= warm:
            draws.append([math.exp(th[0]), th[1]])
    res = RichResult(payload={"draws": draws,
                              "accept_rate": acc / max(tot, 1),
                              "n_draws": len(draws),
                              "method": "random-walk Metropolis GPD posterior (Coles 2001 sec. 9.1.3)"})
    return with_describe_pointer(res, "evbgpd")


def cheatsheet():
    return "evbgpd: Bayesian GPD posterior via Metropolis"


# compact alias per ledger/NAMING.md
evtbayesgpd = evt_bayes_gpd
