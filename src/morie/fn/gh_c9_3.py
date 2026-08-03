# morie.fn -- function file (rootcoder007/morie)
"""Bernstein-polynomial density rate.

Implements sec. 9.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_bpoly_crt"]


def ghosal_bpoly_crt(ns=(100, 800, 6400), seed=42):
    """f(x) = sum_k p_k Be(x; k+1, K-k+1) with K ~ n^{1/3}-ish gives
    rate n^{-s/(2s+1)} up to logs for smooth truths (sec. 9.3).
    Estimates cell weights from counts; L1 error to the triangular
    truth 2x falls with n. Keys: estimate."""
    rng = np.random.default_rng(seed)
    errs = []
    for n in ns:
        K = max(2, int(round(n ** (1.0 / 3.0))))
        data = [math.sqrt(float(rng.uniform(0, 1)))
                for _ in range(n)]                # p0(x) = 2x
        counts = [1.0] * K
        for v in data:
            counts[min(int(v * K), K - 1)] += 1.0
        w = [c / (n + K) for c in counts]
        err = 0.0
        m = 40
        for i in range(m):
            x = (i + 0.5) / m
            dens = sum(
                wk * math.exp(math.lgamma(K + 1.0)
                              - math.lgamma(k + 1.0)
                              - math.lgamma(K - k)
                              + k * math.log(x)
                              + (K - k - 1.0) * math.log(1.0 - x))
                for k, wk in enumerate(w) if 0 < x < 1)
            err += abs(dens - 2.0 * x) / m
        errs.append(err)
    res = RichResult(payload={"estimate": errs[-1],
                              "l1_by_n": errs,
                              "improving": errs[-1] < errs[0],
                              "method": "Bernstein polynomial rate (GvdV 2017 sec. 9.3)"})
    return with_describe_pointer(res, "gh_c9_3")


def cheatsheet():
    return "gh_c9_3: Bernstein-polynomial density rate"
