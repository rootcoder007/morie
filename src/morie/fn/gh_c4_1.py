# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet process definition.

Implements Definition 4.1, eq. (4.1) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_def"]


def ghosal_dp_def(partition_base_masses, seed=42):
    """(P(A_1)..P(A_k)) ~ Dir(k; alpha(A_1)..alpha(A_k)) for every
    finite partition (eq. 4.1): draws the vector by gamma
    normalization. Keys: estimate."""
    a = _bnp._flat(partition_base_masses)
    rng = np.random.default_rng(seed)
    g = [float(rng.gamma(max(ai, 1e-12), 1.0)) for ai in a]
    p = _bnp.normalize_weights(g)
    res = RichResult(payload={"estimate": p[0], "P": p,
                              "dir_params": a,
                              "method": "DP finite-partition law (GvdV 2017 eq. 4.1)"})
    return with_describe_pointer(res, "gh_c4_1")


def cheatsheet():
    return "gh_c4_1: Dirichlet process definition"
