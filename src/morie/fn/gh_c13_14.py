# morie.fn -- function file (rootcoder007/morie)
"""Cox posterior via partial likelihood.

Implements sec. 13.6.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_cox_post"]


def ghosal_cox_post(beta0=0.6, n=400, prior_sd=2.0, seed=42):
    """pi(beta | data) propto pi(beta) prod exp(beta z_i - log
    sum_{j in R_i} exp(beta z_j)) (sec. 13.6.1): grid posterior over
    beta with a normal prior; posterior mean near beta0.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    zs = [1.0 if i % 2 == 0 else 0.0 for i in range(n)]
    times = [-math.log(max(float(rng.uniform(0, 1)), 1e-12))
             / math.exp(beta0 * z) for z in zs]
    order = sorted(range(n), key=lambda i: times[i])
    def log_pl(b):
        tot = -0.5 * (b / prior_sd) ** 2
        risk = sum(math.exp(b * z) for z in zs)
        for idx in order:
            tot += b * zs[idx] - math.log(max(risk, 1e-300))
            risk -= math.exp(b * zs[idx])
        return tot
    grid = [beta0 - 1.5 + 3.0 * j / 60 for j in range(61)]
    ws = [log_pl(b) for b in grid]
    mx = max(ws)
    ws = [math.exp(v - mx) for v in ws]
    Z = sum(ws)
    post_mean = sum(b * w for b, w in zip(grid, ws)) / Z
    res = RichResult(payload={"estimate": post_mean,
                              "error": abs(post_mean - beta0),
                              "method": "Cox posterior (GvdV 2017 sec. 13.6.1)"})
    return with_describe_pointer(res, "gh_c13_14")


def cheatsheet():
    return "gh_c13_14: Cox posterior via partial likelihood"
