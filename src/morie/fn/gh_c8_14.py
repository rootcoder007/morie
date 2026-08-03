# morie.fn -- function file (rootcoder007/morie)
"""Convex-model misspecification.

Implements sec. 8.5.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_convex_misp"]


def ghosal_convex_misp(p0, q1, q2, n_grid=101):
    """For a convex model the KL projection is unique: t ->
    K(p0; (1-t) q1 + t q2) is convex on [0,1] (sec. 8.5.1). Verifies
    convexity along the segment and returns the minimizing mixture.
    Keys: estimate."""
    p0 = _bnp.normalize_weights(p0)
    q1 = _bnp.normalize_weights(q1)
    q2 = _bnp.normalize_weights(q2)
    ts, kls = [], []
    for i in range(n_grid):
        t = i / (n_grid - 1.0)
        q = [(1.0 - t) * a + t * b for a, b in zip(q1, q2)]
        kls.append(sum(x * math.log(x / max(y, 1e-300))
                       for x, y in zip(p0, q) if x > 0))
        ts.append(t)
    # discrete convexity check
    convex = all(kls[i + 1] - 2.0 * kls[i] + kls[i - 1] >= -1e-9
                 for i in range(1, n_grid - 1))
    t_min = ts[kls.index(min(kls))]
    res = RichResult(payload={"estimate": t_min,
                              "kl_min": min(kls),
                              "convex_along_segment": convex,
                              "method": "convex misspecification (GvdV 2017 sec. 8.5.1)"})
    return with_describe_pointer(res, "gh_c8_14")


def cheatsheet():
    return "gh_c8_14: Convex-model misspecification"
