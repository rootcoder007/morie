# morie.fn -- function file (rootcoder007/morie)
"""Regression with DP error distribution.

Implements sec. 7.3.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_regression_posterior"]


def ghosal_dp_regression_posterior(beta0=0.8, n=400, alpha=2.0,
                                   seed=42):
    """Y = f(X) + e, e ~ G ~ DP(alpha, G0) (sec. 7.3.2): with f
    linear, profile beta against the DP-posterior-mean error
    distribution -- E(G(A)|resid) = (M G0(A) + #resid in A)/(M+n)
    (eq. 4.11) evaluated on a residual histogram. Keys: estimate."""
    rng = np.random.default_rng(seed)
    X = [float(rng.uniform(-1, 1)) for _ in range(n)]
    Y = [beta0 * x + 0.5 * float(rng.normal(0, 1)) for x in X]
    k = 16
    def score(b):
        resid = [y - b * x for x, y in zip(X, Y)]
        # DP posterior mean cell masses: (M G0_cell + N_cell)/(M+n)
        counts = [0.0] * k
        for r in resid:
            rr = min(max(r, -3.999), 3.999)
            counts[int((rr + 4.0) / 0.5)] += 1.0
        cell = [(alpha / k + c) / (alpha + n) for c in counts]
        return sum(math.log(cell[int((min(max(r, -3.999), 3.999)
                                      + 4.0) / 0.5)] / 0.5)
                   for r in resid)
    grid = [beta0 - 1.0 + 2.0 * j / 40 for j in range(41)]
    lls = [score(b) for b in grid]
    best = grid[lls.index(max(lls))]
    res = RichResult(payload={"estimate": best,
                              "error": abs(best - beta0),
                              "method": "DP-error regression posterior (GvdV 2017 sec. 7.3.2, eq. 4.11)"})
    return with_describe_pointer(res, "gh_dp_reg_post")


def cheatsheet():
    return "gh_dp_reg_post: Regression with DP error distribution"
