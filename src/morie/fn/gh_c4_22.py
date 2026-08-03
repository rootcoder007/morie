# morie.fn -- function file (rootcoder007/morie)
"""Constrained Dirichlet process.

Implements sec. 4.6.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_constr_dp"]


def ghosal_constr_dp(control_weights, base_masses_by_set, seed=42):
    """DP(alpha) given (P(A_1)..P(A_k)) = w is the finite mixture
    P = sum_j w_j P_j with independent P_j ~ DP(alpha|_{A_j})
    (sec. 4.6.2, via Theorem 4.5 self-similarity). Keys: estimate."""
    w = _bnp._flat(control_weights)
    if abs(sum(w) - 1.0) > 1e-9:
        raise ValueError("control weights must sum to 1")
    rng = np.random.default_rng(seed)
    cells = []
    for wj, masses in zip(w, base_masses_by_set):
        pj = _bnp.normalize_weights(
            [float(rng.gamma(max(a, 1e-12), 1.0))
             for a in _bnp._flat(masses)])
        cells += [wj * v for v in pj]
    res = RichResult(payload={"estimate": cells[0],
                              "P_cells": cells,
                              "total_mass": sum(cells),
                              "method": "constrained DP mixture (GvdV 2017 sec. 4.6.2)"})
    return with_describe_pointer(res, "gh_c4_22")


def cheatsheet():
    return "gh_c4_22: Constrained Dirichlet process"
