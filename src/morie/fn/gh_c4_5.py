# morie.fn -- function file (rootcoder007/morie)
"""DP self-similarity.

Implements Theorem 4.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_selfsim"]


def ghosal_dp_selfsim(w, base_masses_in_A, base_masses_in_Ac,
                      seed=42):
    """Given P(A) = w, the localized processes P_A ~ DP(alpha|_A) and
    P_{A^c} ~ DP(alpha|_{A^c}) are independent DPs (Theorem 4.5):
    P = w P_A + (1-w) P_{A^c}. Returns a draw assembled that way and
    the recombined total mass. Keys: estimate."""
    w = float(w)
    rng = np.random.default_rng(seed)
    pA = _bnp.normalize_weights(
        [float(rng.gamma(max(a, 1e-12), 1.0))
         for a in _bnp._flat(base_masses_in_A)])
    pAc = _bnp.normalize_weights(
        [float(rng.gamma(max(a, 1e-12), 1.0))
         for a in _bnp._flat(base_masses_in_Ac)])
    combined = [w * v for v in pA] + [(1.0 - w) * v for v in pAc]
    res = RichResult(payload={"estimate": combined[0],
                              "P_cells": combined,
                              "total_mass": sum(combined),
                              "method": "DP self-similarity split (GvdV 2017 Thm 4.5)"})
    return with_describe_pointer(res, "gh_c4_5")


def cheatsheet():
    return "gh_c4_5: DP self-similarity"
