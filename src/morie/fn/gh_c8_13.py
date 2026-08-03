# morie.fn -- function file (rootcoder007/morie)
"""Misspecified-model contraction.

Implements sec. 8.5 (contraction at the KL projection) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_misspec_crt"]


def ghosal_misspec_crt(p0=(0.6, 0.3, 0.1), n=2000, seed=42):
    """Under misspecification the posterior contracts to
    P* = argmin_model KL(P0; P) (sec. 8.5). Model: distributions with
    equal first two cells; P* has cells ((p1+p2)/2, (p1+p2)/2, p3).
    Posterior over the model grid concentrates at P*, not P0.
    Keys: estimate."""
    p0 = _bnp.normalize_weights(p0)
    rng = np.random.default_rng(seed)
    counts = [0, 0, 0]
    for _ in range(n):
        u = float(rng.uniform(0, 1))
        counts[0 if u < p0[0] else (1 if u < p0[0] + p0[1]
                                    else 2)] += 1
    # model: p = ((1-t)/2, (1-t)/2, t), t on a grid; flat prior
    grid = 60
    best_t, best_lp = None, -1e18
    for i in range(1, grid):
        t = i / grid
        lp = (counts[0] + counts[1]) * math.log((1.0 - t) / 2.0) \
            + counts[2] * math.log(t)
        if lp > best_lp:
            best_lp, best_t = lp, t
    t_star = p0[2]                        # KL projection: t* = p0_3
    res = RichResult(payload={"estimate": best_t,
                              "kl_projection_t": t_star,
                              "error_to_projection":
                                  abs(best_t - t_star),
                              "method": "misspecified contraction (GvdV 2017 sec. 8.5)"})
    return with_describe_pointer(res, "gh_c8_13")


def cheatsheet():
    return "gh_c8_13: Misspecified-model contraction"
