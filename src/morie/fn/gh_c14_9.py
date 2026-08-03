# morie.fn -- function file (rootcoder007/morie)
"""Pitman-Yor stick-breaking.

Implements sec. 14.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_py_process"]


def ghosal_py_process(n_terms=400, d=0.3, theta=1.0, seed=42):
    """PY(d, theta): V_k ~ Beta(1 - d, theta + k d) (sec. 14.4):
    stick-breaking weights sum to one a.s.; heavier tails than the
    DP. Keys: estimate."""
    rng = np.random.default_rng(seed)
    V = [float(rng.beta(1.0 - d, theta + (k + 1) * d))
         for k in range(n_terms)]
    W = _bnp.stick_breaking(V)
    res = RichResult(payload={"estimate": W[0],
                              "total_mass": sum(W),
                              "top_weights": W[:10],
                              "method": "Pitman-Yor stick breaking (GvdV 2017 sec. 14.4)"})
    return with_describe_pointer(res, "gh_c14_9")


def cheatsheet():
    return "gh_c14_9: Pitman-Yor stick-breaking"
