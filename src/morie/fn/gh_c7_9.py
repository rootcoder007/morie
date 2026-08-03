# morie.fn -- function file (rootcoder007/morie)
"""Linear regression with unknown error.

Implements sec. 7.4.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_linreg_unk_err"]


def ghosal_linreg_unk_err(beta0=1.5, n=400, seed=42):
    """Y = X beta + e with unknown error density (sec. 7.4.2): score
    beta on a grid by a histogram likelihood of the residuals; joint
    consistency for (beta, f) shows as concentration of the profile.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    X = [float(rng.uniform(-1, 1)) for _ in range(n)]
    Y = [beta0 * x + float(rng.normal(0, 0.5)) for x in X]
    def loglik(b):
        resid = [y - b * x for x, y in zip(X, Y)]
        k = 16
        counts = [1.0] * k
        for r in resid:
            rr = min(max(r, -3.999), 3.999)
            counts[int((rr + 4.0) / 0.5)] += 1.0
        tot = sum(counts)
        return sum(math.log(
            counts[int((min(max(r, -3.999), 3.999) + 4.0)
                       / 0.5)] / tot / 0.5) for r in resid)
    grid = [beta0 - 1.0 + 2.0 * j / 40 for j in range(41)]
    lls = [loglik(b) for b in grid]
    best = grid[lls.index(max(lls))]
    res = RichResult(payload={"estimate": best,
                              "error": abs(best - beta0),
                              "method": "regression, unknown error law (GvdV 2017 sec. 7.4.2)"})
    return with_describe_pointer(res, "gh_c7_9")


def cheatsheet():
    return "gh_c7_9: Linear regression with unknown error"
