# morie.fn -- function file (rootcoder007/morie)
"""Bayesian bootstrap.

Implements Corollary 4.17(ii) + sec. 4.7 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_bayes_boot"]


def ghosal_bayes_boot(data, n_draws=200, seed=42):
    """|alpha| -> 0 gives DP(sum delta_Xi): weights (W_1..W_n) ~
    Dir(n; 1..1), realized as W_i = Y_i / sum Y_j with Y_i iid Exp(1)
    (sec. 4.7). Returns Monte Carlo draws of the mean functional.
    Keys: estimate."""
    xs = _bnp._flat(data)
    n = len(xs)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(n_draws)):
        Y = [-math.log(max(float(rng.uniform(0, 1)), 1e-300))
             for _ in range(n)]
        tot = sum(Y)
        means.append(sum(y / tot * x for y, x in zip(Y, xs)))
    est = sum(means) / len(means)
    res = RichResult(payload={"estimate": est,
                              "draws_head": means[:10],
                              "sample_mean": sum(xs) / n,
                              "method": "Bayesian bootstrap (GvdV 2017 sec. 4.7)"})
    return with_describe_pointer(res, "gh_c4_24")


def cheatsheet():
    return "gh_c4_24: Bayesian bootstrap"
