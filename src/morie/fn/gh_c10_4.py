# morie.fn -- function file (rootcoder007/morie)
"""Two-model adaptation.

Implements sec. 10.2.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_two_model_adp"]


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


def ghosal_two_model_adp(n=500, truth_dim=1, pi0=0.5, seed=42):
    """Pi = pi_0 Pi_0 + (1 - pi_0) Pi_1: the posterior weight of the
    model containing the truth tends to one (sec. 10.2.3). Model 0
    keeps 1 coordinate, model 1 keeps 6; exact evidence ratio.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    y = [(1.0 if k < truth_dim else 0.0)
         + float(rng.normal(0, 1)) / math.sqrt(n) for k in range(6)]
    l0 = _log_evidence_K(y, n, 1) + math.log(pi0)
    l1 = _log_evidence_K(y, n, 6) + math.log(1.0 - pi0)
    w0 = 1.0 / (1.0 + math.exp(l1 - l0))
    res = RichResult(payload={"estimate": w0,
                              "posterior_weight_small_model": w0,
                              "small_model_wins": w0 > 0.5,
                              "method": "two-model adaptation (GvdV 2017 sec. 10.2.3)"})
    return with_describe_pointer(res, "gh_c10_4")


def cheatsheet():
    return "gh_c10_4: Two-model adaptation"
