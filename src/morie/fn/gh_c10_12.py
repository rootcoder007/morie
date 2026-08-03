# morie.fn -- function file (rootcoder007/morie)
"""Bayes-factor model selection consistency.

Implements sec. 10.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_modsel_bic"]


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


def ghosal_modsel_bic(truth_in_H1=True, n=2000, seed=42):
    """BF(H1, H0) = p(X | H1)/p(X | H0) tends to infinity under H1
    and to 0 under H0 (sec. 10.5). Exact normal evidences: H0 keeps
    0 coordinates, H1 keeps 2. Keys: estimate."""
    rng = np.random.default_rng(seed)
    mu = 0.7 if truth_in_H1 else 0.0
    y = [mu + float(rng.normal(0, 1)) / math.sqrt(n)
         for _ in range(2)]
    log_bf = _log_evidence_K(y, n, 2) - _log_evidence_K(y, n, 0)
    res = RichResult(payload={"estimate": log_bf,
                              "supports_H1": log_bf > 0,
                              "method": "Bayes factor consistency (GvdV 2017 sec. 10.5)"})
    return with_describe_pointer(res, "gh_c10_12")


def cheatsheet():
    return "gh_c10_12: Bayes-factor model selection consistency"
