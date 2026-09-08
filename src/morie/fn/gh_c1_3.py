# morie.fn -- function file (rootcoder007/morie)
"""Sequential prior-to-posterior updating.

Implements sec. 1.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_prior_posterior_update"]


def ghosal_prior_posterior_update(x, data=None, log_lik_one=None,
                                  log_prior=None):
    """dPi_n/dPi(theta) = p_theta(X^n) / int p_eta(X^n) dPi(eta)
    (GvdV 2017 sec. 1.3): batch updating equals sequential updating.
    Verified internally: the posterior from all data at once matches
    updating one observation at a time."""
    import math
    th = _bnp._flat(x)
    if data is None:
        data = [0.8, 1.2, 1.0]
    if log_lik_one is None:
        log_lik_one = lambda t, d: -0.5 * (d - t) ** 2
    if log_prior is None:
        log_prior = lambda t: -0.5 * t * t
    # batch
    lw = [sum(log_lik_one(t, d) for d in data) + log_prior(t)
          for t in th]
    mx = max(lw)
    w = [math.exp(v - mx) for v in lw]
    tot = sum(w)
    post = [v / tot for v in w]
    # sequential
    logp = [log_prior(t) for t in th]
    for d in data:
        logp = [lp + log_lik_one(t, d) for lp, t in zip(logp, th)]
    mx2 = max(logp)
    w2 = [math.exp(v - mx2) for v in logp]
    tot2 = sum(w2)
    seq = [v / tot2 for v in w2]
    drift = max(abs(a - b) for a, b in zip(post, seq))
    est = sum(t * p for t, p in zip(th, post))
    res = RichResult(payload={"estimate": est, "posterior": post,
                              "sequential_batch_gap": drift,
                              "method": "prior-to-posterior updating (GvdV 2017 sec. 1.3)"})
    return with_describe_pointer(res, "gh_c1_3")


def cheatsheet():
    return "gh_c1_3: Sequential prior-to-posterior updating"
