# morie.fn -- function file (rootcoder007/morie)
"""Adaptation to a parametric truth.

Implements sec. 10.2.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_param_rate"]


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


def ghosal_param_rate(d_true=2, ns=(100, 1000, 10000), lam=1.0,
                      seed=42):
    """If f0 sits in a d-dimensional submodel, the hierarchical
    posterior attains the parametric sqrt(d/n) rate (sec. 10.2.2):
    the selected dimension stays near d and the risk scales like
    d/n. Keys: estimate."""
    rng = np.random.default_rng(seed)
    risks = []
    for n in ns:
        y = [(0.8 if k < d_true else 0.0)
             + float(rng.normal(0, 1)) / math.sqrt(n)
             for k in range(10)]
        logs = [_log_evidence_K(y, n, K) - lam * K * math.log(n)
                for K in range(11)]
        k_hat = logs.index(max(logs))
        # posterior risk of the selected model: sum of posterior vars
        # + squared bias of zeroed coords
        risk = k_hat * (1.0 / (n + 1.0)) \
            + sum(0.64 for k in range(k_hat, d_true))
        risks.append(risk)
    rate_hat = math.log(risks[0] / risks[-1]) \
        / math.log(float(ns[-1]) / ns[0])
    res = RichResult(payload={"estimate": rate_hat,
                              "risk_by_n": risks,
                              "parametric": abs(rate_hat - 1.0)
                              < 0.25,
                              "method": "parametric adaptation (GvdV 2017 sec. 10.2.2)"})
    return with_describe_pointer(res, "gh_c10_3")


def cheatsheet():
    return "gh_c10_3: Adaptation to a parametric truth"
