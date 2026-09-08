# morie.fn -- function file (rootcoder007/morie)
"""Predictive (Cesàro) consistency.

Implements Theorem 6.50, eq. (6.13) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_pred_consist"]


def ghosal_pred_consist(theta0=0.35, n=800, seed=42):
    """n^{-1} sum_i K(p0; p-hat_i) -> 0 in mean when p0 in KL(Pi)
    (Thm 6.50): Beta-Bernoulli predictive p-hat_i is the posterior
    mean before the i-th observation; the Cesaro-averaged KL to the
    truth decays. Keys: estimate."""
    rng = np.random.default_rng(seed)
    S = 0
    tot_kl = 0.0
    path = []
    for i in range(n):
        pred = (1.0 + S) / (2.0 + i)
        tot_kl += theta0 * math.log(theta0 / pred) \
            + (1 - theta0) * math.log((1 - theta0) / (1 - pred))
        if (i + 1) % (n // 8) == 0:
            path.append(tot_kl / (i + 1))
        S += 1 if float(rng.uniform(0, 1)) < theta0 else 0
    res = RichResult(payload={"estimate": path[-1],
                              "cesaro_kl_path": path,
                              "decaying": path[-1] < path[0],
                              "method": "predictive Cesaro consistency (GvdV 2017 Thm 6.50)"})
    return with_describe_pointer(res, "gh_c6_14")


def cheatsheet():
    return "gh_c6_14: Predictive (Cesàro) consistency"
