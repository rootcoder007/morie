# morie.fn -- function file (rootcoder007/morie)
"""Strong posterior consistency.

Implements Definition 6.1 (almost-sure form) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_strong_consist"]


def ghosal_strong_consist(theta0=0.6, eps=0.15, n=2000, seed=42):
    """Strong consistency: along a single sample path the posterior
    mass outside the ball tends to 0 (Def 6.1, a.s. sense). Tracks
    the exact Beta posterior tail along one Bernoulli path.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    S = 0
    checkpoints = {}
    for i in range(1, n + 1):
        S += 1 if float(rng.uniform(0, 1)) < theta0 else 0
        if i in (50, 200, 800, n):
            a, b = 1.0 + S, 1.0 + i - S
            grid = 2000
            mass = 0.0
            for k in range(grid):
                t = (k + 0.5) / grid
                if abs(t - theta0) > eps:
                    mass += math.exp(
                        math.lgamma(a + b) - math.lgamma(a)
                        - math.lgamma(b)
                        + (a - 1.0) * math.log(t)
                        + (b - 1.0) * math.log(1.0 - t)) / grid
            checkpoints[i] = mass
    ks = sorted(checkpoints)
    res = RichResult(payload={"estimate": checkpoints[ks[-1]],
                              "path_masses": [checkpoints[k]
                                              for k in ks],
                              "method": "strong consistency along a path (GvdV 2017 Def 6.1)"})
    return with_describe_pointer(res, "gh_c6_2")


def cheatsheet():
    return "gh_c6_2: Strong posterior consistency"
