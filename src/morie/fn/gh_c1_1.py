# morie.fn -- function file (rootcoder007/morie)
"""Bayes's rule with a dominated likelihood.

Implements sec. 1.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_bayes_rule_infinite"]


def ghosal_bayes_rule_infinite(x, log_lik=None, log_prior=None):
    """Posterior on a grid: pi(theta|X) proportional to
    p_theta(X) pi(theta) (GvdV 2017 sec. 1.3). ``x`` is a grid of
    parameter values; densities default to a conjugate demonstration
    (standard-normal prior, N(theta,1) likelihood of one observation
    at 1.0) so the module is runnable stand-alone."""
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
    post = [v / tot for v in w]
    est = sum(t * p for t, p in zip(th, post))
    res = RichResult(payload={"estimate": est, "posterior": post,
                              "grid": th,
                              "method": "Bayes rule on a grid (GvdV 2017 sec. 1.3)"})
    return with_describe_pointer(res, "gh_c1_1")


def cheatsheet():
    return "gh_c1_1: Bayes's rule with a dominated likelihood"
