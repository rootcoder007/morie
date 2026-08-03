# morie.fn -- function file (rootcoder007/morie)
"""Interval censoring with a DP prior.

Implements sec. 9.5.7 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_icens_dp_crt"]


def ghosal_icens_dp_crt(ns=(200, 1600, 12800), seed=42):
    """Current-status data: observe (C_i, 1{X_i <= C_i}); the DP/NPMLE
    posterior for monotone F contracts at (n/log n)^{-1/3}
    (sec. 9.5.7). Binned Beta posterior + isotonic projection; error
    to F0(x) = x falls at roughly the cube-root rate.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    errs = []
    for n in ns:
        k = max(4, int(round(n ** (1.0 / 3.0))))
        succ = [0.0] * k
        tot = [0.0] * k
        for _ in range(n):
            xv = float(rng.uniform(0, 1))
            cv = float(rng.uniform(0, 1))
            b = min(int(cv * k), k - 1)
            succ[b] += 1.0 if xv <= cv else 0.0
            tot[b] += 1.0
        vals = [(1.0 + s) / (2.0 + t) for s, t in zip(succ, tot)]
        wts = [2.0 + t for t in tot]
        i = 0
        while i < len(vals) - 1:               # PAVA
            if vals[i] > vals[i + 1] + 1e-12:
                wm = wts[i] + wts[i + 1]
                vm = (vals[i] * wts[i]
                      + vals[i + 1] * wts[i + 1]) / wm
                vals[i:i + 2] = [vm]
                wts[i:i + 2] = [wm]
                i = max(i - 1, 0)
            else:
                i += 1
        # expand isotonic blocks back to k cells by weight
        F = []
        for v, w in zip(vals, wts):
            reps = max(int(round(w / (n / k + 2.0))), 1)
            F += [v] * reps
        F = F[:k] + [vals[-1]] * max(0, k - len(F))
        errs.append(sum(abs(F[b] - (b + 0.5) / k)
                        for b in range(k)) / k)
    res = RichResult(payload={"estimate": errs[-1],
                              "err_by_n": errs,
                              "improving": errs[-1] < errs[0],
                              "method": "interval-censored DP rate (GvdV 2017 sec. 9.5.7)"})
    return with_describe_pointer(res, "gh_c9_11")


def cheatsheet():
    return "gh_c9_11: Interval censoring with a DP prior"
