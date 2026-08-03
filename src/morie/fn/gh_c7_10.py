# morie.fn -- function file (rootcoder007/morie)
"""Monotone binary regression.

Implements sec. 7.4.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_mono_reg_con"]


def ghosal_mono_reg_con(n=600, seed=42):
    """P(Y=1|x) = F(x) monotone with a DP-type prior on F
    (sec. 7.4.3): cellwise Beta posterior means followed by isotonic
    (pool-adjacent-violators) projection stay consistent in the weak
    topology. Truth F(x) = x on [0,1]. Keys: estimate."""
    rng = np.random.default_rng(seed)
    k = 10
    succ = [0.0] * k
    tot = [0.0] * k
    for _ in range(n):
        x = float(rng.uniform(0, 1))
        y = 1.0 if float(rng.uniform(0, 1)) < x else 0.0
        c = min(int(x * k), k - 1)
        succ[c] += y
        tot[c] += 1.0
    post = [(1.0 + s) / (2.0 + t) for s, t in zip(succ, tot)]
    # PAVA
    vals = post[:]
    wts = [t + 2.0 for t in tot]
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-12:
            wm = wts[i] + wts[i + 1]
            vm = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / wm
            vals[i:i + 2] = [vm]
            wts[i:i + 2] = [wm]
            i = max(i - 1, 0)
        else:
            i += 1
    # expand back
    F = []
    for v, w in zip(vals, wts):
        F += [v] * max(int(round(w / (n / k + 2.0))), 1)
    F = F[:k] + [vals[-1]] * max(0, k - len(F))
    truth = [(c + 0.5) / k for c in range(k)]
    err = max(abs(a - b) for a, b in zip(F, truth))
    res = RichResult(payload={"estimate": err,
                              "F_cells": F,
                              "monotone": all(F[i] <= F[i + 1] + 1e-9
                                              for i in range(k - 1)),
                              "method": "monotone binary regression (GvdV 2017 sec. 7.4.3)"})
    return with_describe_pointer(res, "gh_c7_10")


def cheatsheet():
    return "gh_c7_10: Monotone binary regression"
