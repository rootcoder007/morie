# morie.fn -- function file (rootcoder007/morie)
"""Integrated Brownian motion.

Implements Example 11.6, eq. (11.1) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gp_brownian_primitive"]


def ghosal_gp_brownian_primitive(k=1, n_grid=256, seed=42):
    """I_{0+}^k B: k-fold primitive of BM has C^{k-1} paths with
    nearly-Lipschitz-1/2 k-th derivative -- smoothness k + 1/2
    (Ex 11.6, eq. 11.1). Path roughness (mean |increment|) drops by
    a factor ~ n_grid after each integration. Keys: estimate."""
    rng = np.random.default_rng(seed)
    path = []
    w = 0.0
    for _ in range(n_grid):
        w += float(rng.normal(0, 1)) / math.sqrt(n_grid)
        path.append(w)
    def rough(p):
        return sum(abs(p[i + 1] - p[i]) for i in range(len(p) - 1)) \
            / (len(p) - 1)
    r0 = rough(path)
    cur = path
    for _ in range(int(k)):
        acc = 0.0
        out = []
        for v in cur:
            acc += v / n_grid
            out.append(acc)
        cur = out
    rk = rough(cur)
    res = RichResult(payload={"estimate": rk,
                              "roughness_bm": r0,
                              "smoother": rk < r0 / 10.0,
                              "method": "integrated BM primitive (GvdV 2017 Ex 11.6)"})
    return with_describe_pointer(res, "gh_gp_brow_prim")


def cheatsheet():
    return "gh_gp_brow_prim: Integrated Brownian motion"
