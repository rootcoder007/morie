# morie.fn -- function file (rootcoder007/morie)
"""Beta-process definition.

Implements sec. 13.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_beta_proc_def"]


def ghosal_beta_proc_def(grid_t, c=2.0, Lambda0_rate=1.0, seed=42):
    """BP(c, H0): increments dH(t) ~ Be(c dH0(t), c(1 - dH0(t)))
    independently (sec. 13.3): simulates the cumulative hazard on a
    grid with H0 the unit-exponential cumulative hazard.
    Keys: estimate."""
    ts = _bnp._flat(grid_t)
    rng = np.random.default_rng(seed)
    H = 0.0
    path = []
    prev = 0.0
    for t in ts:
        dH0 = Lambda0_rate * (t - prev)
        a = max(c * dH0, 1e-8)
        b = max(c * (1.0 - dH0), 1e-8)
        H += float(rng.beta(a, b))
        path.append(H)
        prev = t
    res = RichResult(payload={"estimate": path[-1],
                              "cum_hazard": path,
                              "nondecreasing": all(
                                  path[i + 1] >= path[i] - 1e-12
                                  for i in range(len(path) - 1)),
                              "method": "beta process (GvdV 2017 sec. 13.3)"})
    return with_describe_pointer(res, "gh_c13_3")


def cheatsheet():
    return "gh_c13_3: Beta-process definition"
