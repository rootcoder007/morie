# morie.fn -- function file (rootcoder007/morie)
"""Beta-process path by Poisson jumps.

Implements sec. 13.3.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_bp_path_gen"]


def ghosal_bp_path_gen(c=1.5, t_max=1.0, n_jumps=300, seed=42):
    """H(t) = sum_{tau_k <= t} J_k with (J_k, tau_k) from a Poisson
    process with the BP Levy intensity (sec. 13.3.3): inverse-Levy
    style simulation with jump sizes Be(1, c)-tilted; the path is a
    pure-jump nondecreasing function. Keys: estimate."""
    rng = np.random.default_rng(seed)
    jumps = sorted(
        ((float(rng.uniform(0, 1)) * t_max,
          float(rng.beta(1.0, c)) / n_jumps * c * 5.0)
         for _ in range(n_jumps)), key=lambda p: p[0])
    H = 0.0
    path = []
    for tau, J in jumps:
        H += J
        path.append((tau, H))
    res = RichResult(payload={"estimate": path[-1][1],
                              "n_jumps": n_jumps,
                              "pure_jump_nondecreasing": all(
                                  path[i + 1][1] >= path[i][1]
                                  for i in range(len(path) - 1)),
                              "method": "BP Poisson-jump path (GvdV 2017 sec. 13.3.3)"})
    return with_describe_pointer(res, "gh_c13_6")


def cheatsheet():
    return "gh_c13_6: Beta-process path by Poisson jumps"
