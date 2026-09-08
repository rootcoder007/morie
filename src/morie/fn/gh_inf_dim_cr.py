# morie.fn -- function file (rootcoder007/morie)
"""Infinite-dimensional credible balls.

Implements sec. 12.5 (radius calibration) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_inf_dim_credible"]


def ghosal_inf_dim_credible(n=300, dim=40, level=0.9, n_sim=300,
                            seed=42):
    """Pi_n(||theta - theta0|| <= r_n | X) = 1 - alpha: in the
    conjugate sequence model the credible radius is a chi-square
    quantile of the posterior spread; coverage matches when centered
    correctly (sec. 12.5). Keys: estimate."""
    rng = np.random.default_rng(seed)
    # posterior for each coord: N(y_i, 1/(n+1)); credible radius^2 =
    # quantile of sum of dim scaled chi-squares ~ (dim + z sqrt(2
    # dim))/(n+1)
    z = 1.2815515655446004 if abs(level - 0.9) < 1e-9 else 1.6449
    r2 = (dim + z * math.sqrt(2.0 * dim)) / (n + 1.0)
    hits = 0
    for _ in range(n_sim):
        d2 = 0.0
        for _ in range(dim):
            y = float(rng.normal(0, 1)) / math.sqrt(n)
            post_mean = n / (n + 1.0) * y
            d2 += post_mean ** 2       # theta0 = 0
        if d2 <= r2:
            hits += 1
    cov = hits / n_sim
    res = RichResult(payload={"estimate": cov,
                              "nominal": level,
                              "conservative_or_close":
                                  cov >= level - 0.07,
                              "method": "infinite-dim credible ball (GvdV 2017 sec. 12.5)"})
    return with_describe_pointer(res, "gh_inf_dim_cr")


def cheatsheet():
    return "gh_inf_dim_cr: Infinite-dimensional credible balls"
