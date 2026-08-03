# morie.fn -- function file (rootcoder007/morie)
"""Cox-model BvM.

Implements sec. 12.3.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_cox_bvm_sp"]


def ghosal_cox_bvm_sp(beta0=0.8, n=600, seed=42):
    """sqrt(n)(beta_post - beta0) -> N(0, I_beta^{-1}) via the
    partial likelihood (sec. 12.3.3). Exponential baseline, binary
    covariate: the partial-likelihood MAP recovers beta0.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    zs = [1.0 if i % 2 == 0 else 0.0 for i in range(n)]
    times = [-math.log(max(float(rng.uniform(0, 1)), 1e-12))
             / math.exp(beta0 * z) for z in zs]
    order = sorted(range(n), key=lambda i: times[i])
    def neg_pll(b):
        # partial likelihood over event ordering
        tot = 0.0
        risk = sum(math.exp(b * z) for z in zs)
        for idx in order:
            tot -= b * zs[idx] - math.log(max(risk, 1e-300))
            risk -= math.exp(b * zs[idx])
        return tot
    grid = [beta0 - 1.0 + 2.0 * j / 50 for j in range(51)]
    vals = [neg_pll(b) for b in grid]
    b_hat = grid[vals.index(min(vals))]
    res = RichResult(payload={"estimate": b_hat,
                              "error": abs(b_hat - beta0),
                              "method": "Cox partial-likelihood BvM (GvdV 2017 sec. 12.3.3)"})
    return with_describe_pointer(res, "gh_c12_8")


def cheatsheet():
    return "gh_c12_8: Cox-model BvM"
