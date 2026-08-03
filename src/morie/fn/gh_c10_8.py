# morie.fn -- function file (rootcoder007/morie)
"""Finite-random-series regression.

Implements sec. 10.4.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_frs_reg"]


def ghosal_frs_reg(n=600, seed=42):
    """Y_i = f(x_i) + e_i, f = sum_{k<=K} beta_k phi_k: choosing K by
    conjugate evidence gives rate n^{-2s/(2s+1)} (sec. 10.4.2).
    Cosine-basis ridge fits per K; evidence picks K; reports risk.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    xs = [(i + 0.5) / n for i in range(n)]
    f0 = [math.sin(2.0 * math.pi * x) for x in xs]
    ys = [f + 0.4 * float(rng.normal(0, 1)) for f in f0]
    def phi(x, k):
        return math.sqrt(2.0) * math.cos((k + 1) * math.pi * x)
    best = None
    for K in range(1, 9):
        coef = []
        for k in range(K):
            num = sum(y * phi(x, k) for x, y in zip(xs, ys)) / n
            coef.append(num * n / (n + 1.0))     # ridge w/ unit prior
        rss = 0.0
        for x, y in zip(xs, ys):
            fx = sum(c * phi(x, k) for k, c in enumerate(coef))
            rss += (y - fx) ** 2
        # Laplace-type evidence: -n/2 log(rss/n) - K/2 log n
        ev = -0.5 * n * math.log(rss / n) - 0.5 * K * math.log(n)
        if best is None or ev > best[0]:
            best = (ev, K, coef)
    _, K_hat, coef = best
    risk = sum((sum(c * phi(x, k) for k, c in enumerate(coef))
                - f) ** 2 for x, f in zip(xs, f0)) / n
    res = RichResult(payload={"estimate": risk, "K_hat": K_hat,
                              "method": "finite random series regression (GvdV 2017 sec. 10.4.2)"})
    return with_describe_pointer(res, "gh_c10_8")


def cheatsheet():
    return "gh_c10_8: Finite-random-series regression"
