# morie.fn -- function file (rootcoder007/morie)
"""Semiparametric location.

Implements sec. 7.4.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_loc_semipara"]


def ghosal_loc_semipara(theta0=0.7, n=400, seed=42):
    """X_i = theta + e_i with unknown symmetric error density
    (sec. 7.4.1): profile the location over a grid, scoring each
    theta by a symmetrized histogram likelihood of the residuals;
    the posterior mode concentrates at theta0. Keys: estimate."""
    rng = np.random.default_rng(seed)
    data = [theta0 + float(rng.normal(0, 1)) for _ in range(n)]
    def loglik(th):
        resid = [x - th for x in data]
        # symmetrized histogram on [-4, 4], 16 cells
        k = 16
        counts = [1.0] * k               # +1 smoothing (Dirichlet)
        for r in resid:
            rr = min(max(r, -3.999), 3.999)
            idx = int((rr + 4.0) / 0.5)
            counts[idx] += 0.5
            counts[k - 1 - idx] += 0.5   # symmetrize
        tot = sum(counts)
        ll = 0.0
        for r in resid:
            rr = min(max(r, -3.999), 3.999)
            idx = int((rr + 4.0) / 0.5)
            ll += math.log(counts[idx] / tot / 0.5)
        return ll
    grid = [theta0 - 1.0 + 2.0 * j / 40 for j in range(41)]
    lls = [loglik(t) for t in grid]
    best = grid[lls.index(max(lls))]
    res = RichResult(payload={"estimate": best,
                              "error": abs(best - theta0),
                              "method": "semiparametric location (GvdV 2017 sec. 7.4.1)"})
    return with_describe_pointer(res, "gh_c7_8")


def cheatsheet():
    return "gh_c7_8: Semiparametric location"
