# morie.fn -- function file (rootcoder007/morie)
"""Random-series prior with random dimension.

Implements sec. 10.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_rnd_series_pr"]


def _log_evidence_K(y, n_prec, K, tau2=1.0):
    """Exact log marginal likelihood of a normal-means model that
    keeps the first K coordinates (theta_k ~ N(0, tau2)) and zeroes
    the rest; y_k | theta ~ N(theta_k, 1/n_prec)."""
    lp = 0.0
    v = 1.0 / n_prec
    for k, yk in enumerate(y):
        s2 = v + (tau2 if k < K else 0.0)
        lp += -0.5 * math.log(2.0 * math.pi * s2) \
            - 0.5 * yk * yk / s2
    return lp


def ghosal_rnd_series_pr(K_true=4, n=1000, lam=0.5, K_max=15,
                         seed=42):
    """K ~ pi_n (geometric-type), beta_k | K iid N(0, sigma^2): the
    posterior on K concentrates near the effective dimension and the
    rate adapts to smoothness (sec. 10.4). Exact evidence per K.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    y = [(0.9 if k < K_true else 0.0)
         + float(rng.normal(0, 1)) / math.sqrt(n)
         for k in range(K_max)]
    logs = [_log_evidence_K(y, n, K) - lam * K * math.log(n)
            for K in range(K_max + 1)]
    mx = max(logs)
    w = [math.exp(v - mx) for v in logs]
    Z = sum(w)
    post = [v / Z for v in w]
    mean_K = sum(k * p for k, p in enumerate(post))
    res = RichResult(payload={"estimate": mean_K,
                              "K_posterior": post,
                              "mode_K": post.index(max(post)),
                              "method": "random series prior (GvdV 2017 sec. 10.4)"})
    return with_describe_pointer(res, "gh_c10_6")


def cheatsheet():
    return "gh_c10_6: Random-series prior with random dimension"
