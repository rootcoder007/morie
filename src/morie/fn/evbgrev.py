# morie.fn -- function file (rootcoder007/morie)
"""Bayesian GEV via Metropolis with prior π(μ,σ,ξ).

Implements sec. 9.1.2-9.1.3 (MCMC for the GEV posterior) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer (equation checked against the
library PDF).
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_bayes_gev"]


def evt_bayes_gev(x, n_draws=2000, seed=42, prior_sd=(100.0, 10.0, 1.0)):
    """Random-walk Metropolis on (mu, log sigma, xi) targeting the GEV
    posterior with independent mean-zero normal priors of the given
    standard deviations (the vague-prior construction of Coles 2001
    sec. 9.1.3). Step sizes adapt during warm-up toward the ~35%
    acceptance heuristic."""
    import math
    xs = _ev._flat(x)
    f = _ev.gev_mle(xs)
    rng = np.random.default_rng(seed)
    s_mu, s_ls, s_xi = prior_sd

    def logpost(mu, ls, xi):
        lp = _ev.gev_loglik(xs, mu, math.exp(ls), xi)
        lp += -mu * mu / (2 * s_mu * s_mu)
        lp += -ls * ls / (2 * s_ls * s_ls)
        lp += -xi * xi / (2 * s_xi * s_xi)
        return lp

    th = [f["mu"], math.log(f["sigma"]), f["xi"]]
    step = [0.1, 0.05, 0.05]
    lp = logpost(*th)
    warm = max(200, int(n_draws) // 2)
    draws = []
    acc = tot = 0
    for it in range(warm + int(n_draws)):
        for j in range(3):
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
            draws.append([th[0], math.exp(th[1]), th[2]])
    res = RichResult(payload={"draws": draws,
                              "accept_rate": acc / max(tot, 1),
                              "n_draws": len(draws),
                              "method": "random-walk Metropolis GEV posterior (Coles 2001 sec. 9.1.3)"})
    return with_describe_pointer(res, "evbgrev")


def cheatsheet():
    return "evbgrev: Bayesian GEV via Metropolis with prior π(μ,σ,ξ)"
