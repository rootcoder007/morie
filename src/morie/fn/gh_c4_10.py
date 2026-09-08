# morie.fn -- function file (rootcoder007/morie)
"""Pólya-sequence construction of the DP.

Implements sec. 4.2.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_polya_urn"]


def ghosal_dp_polya_urn(n, alpha, seed=42):
    """X_1 ~ G0-bar; X_{n+1} | X ~ (alpha + sum delta_Xi)/(|alpha|+n)
    (sec. 4.2.4): simulates the Polya sequence with uniform center
    measure and reports the tie structure. Keys: estimate."""
    M = float(alpha)
    rng = np.random.default_rng(seed)
    xs = []
    for i in range(int(n)):
        if not xs or float(rng.uniform(0, 1)) < M / (M + len(xs)):
            xs.append(float(rng.uniform(0, 1)))       # fresh from G0
        else:
            j = int(float(rng.uniform(0, 1)) * len(xs))
            xs.append(xs[min(j, len(xs) - 1)])        # copy old value
    k = len(set(xs))
    res = RichResult(payload={"estimate": float(k), "n": int(n),
                              "n_distinct": k, "draws_head": xs[:10],
                              "method": "Polya urn sequence (GvdV 2017 sec. 4.2.4)"})
    return with_describe_pointer(res, "gh_c4_10")


def cheatsheet():
    return "gh_c4_10: Pólya-sequence construction of the DP"
