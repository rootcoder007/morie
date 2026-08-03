# morie.fn -- function file (rootcoder007/morie)
"""Countable Dirichlet process.

Implements sec. 3.3.3, eq. (3.4) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_countable_dp"]


def ghosal_countable_dp(x, alpha_total=5.0, k=6, seed=42):
    """(p_1, ..., p_k, 1-sum) ~ Dir(k+1; alpha_1..alpha_k, tail)
    (GvdV 2017 eq. 3.4), realized by gamma normalization with a tail
    cell carrying the remaining concentration."""
    rng = np.random.default_rng(seed)
    a = [alpha_total / (2.0 ** (j + 1)) for j in range(k)]
    tail = alpha_total - sum(a)
    g = [float(rng.gamma(max(ai, 1e-8), 1.0)) for ai in a + [tail]]
    p = _bnp.normalize_weights(g)
    res = RichResult(payload={"estimate": p[0], "p_cells": p[:k],
                              "p_tail": p[k], "alpha": a,
                              "alpha_tail": tail,
                              "method": "countable Dirichlet process (GvdV 2017 eq. 3.4)"})
    return with_describe_pointer(res, "gh_c3_5")


def cheatsheet():
    return "gh_c3_5: Countable Dirichlet process"
