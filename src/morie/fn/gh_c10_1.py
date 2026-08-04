# morie.fn -- function file (rootcoder007/morie)
"""Adaptive hierarchical model prior.

Implements sec. 10.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_adapt_thm"]


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


def ghosal_adapt_thm(y=None, n=200, K_true=3, lam=1.0, K_max=12,
                     seed=42):
    """Pi = sum_k pi_k Pi_k with pi_k propto exp(-lam k log n): the
    posterior over model index k concentrates and the rate adapts to
    the truth (sec. 10.1). Exact conjugate evidence per k plus the
    complexity prior gives the model posterior. Keys: estimate."""
    rng = np.random.default_rng(seed)
    if y is None:
        y = [(1.0 if k < K_true else 0.0)
             + float(rng.normal(0, 1)) / math.sqrt(n)
             for k in range(K_max)]
    logs = [_log_evidence_K(y, n, K)
            - lam * K * math.log(n) for K in range(K_max + 1)]
    mx = max(logs)
    w = [math.exp(v - mx) for v in logs]
    tot = sum(w)
    post = [v / tot for v in w]
    k_hat = post.index(max(post))
    res = RichResult(payload={"estimate": float(k_hat),
                              "model_posterior": post,
                              "K_true": K_true,
                              "method": "adaptive model prior (GvdV 2017 sec. 10.1)"})
    return with_describe_pointer(res, "gh_c10_1")


def cheatsheet():
    return "gh_c10_1: Adaptive hierarchical model prior"


# compact alias per ledger/NAMING.md
ghosaladaptthm = ghosal_adapt_thm
