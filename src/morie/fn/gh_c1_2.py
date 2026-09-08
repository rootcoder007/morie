# morie.fn -- function file (rootcoder007/morie)
"""Posterior via Radon–Nikodym under domination.

Implements sec. 1.3.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_absolute_continuity"]


def ghosal_absolute_continuity(x, log_lik=None, log_prior=None):
    """When every P_theta has a density w.r.t. a common mu, the
    posterior exists and is the normalized density-weighted prior
    (GvdV 2017 sec. 1.3.1). Returns the normalizing constant (the
    marginal likelihood) alongside the posterior; a zero marginal
    would mean the domination assumption failed on this grid."""
    import math
    th = _bnp._flat(x)
    if log_lik is None:
        log_lik = lambda t: -0.5 * (1.0 - t) ** 2
    if log_prior is None:
        log_prior = lambda t: -0.5 * t * t
    lw = [log_lik(t) + log_prior(t) for t in th]
    mx = max(lw)
    w = [math.exp(v - mx) for v in lw]
    tot = sum(w)
    marg = math.log(tot / len(th)) + mx
    post = [v / tot for v in w]
    est = sum(t * p for t, p in zip(th, post))
    res = RichResult(payload={"estimate": est,
                              "log_marginal": marg,
                              "posterior": post,
                              "method": "dominated posterior, Radon-Nikodym (GvdV 2017 sec. 1.3.1)"})
    return with_describe_pointer(res, "gh_c1_2")


def cheatsheet():
    return "gh_c1_2: Posterior via Radon–Nikodym under domination"
