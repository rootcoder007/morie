# morie.fn -- function file (rootcoder007/morie)
"""Mutual singularity of Dirichlet processes.

Implements Theorem 4.21 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_mutual_sing"]


def ghosal_dp_mutual_sing(cont_part_1, cont_part_2, atoms_1, atoms_2):
    """DP(alpha_1) and DP(alpha_2) are mutually singular if the
    continuous parts differ or the atomic supports differ
    (Theorem 4.21). Continuous parts given as (mass on a grid);
    atoms as support-point lists. Keys: estimate."""
    c1 = _bnp._flat(cont_part_1)
    c2 = _bnp._flat(cont_part_2)
    a1 = set(_bnp._flat(atoms_1))
    a2 = set(_bnp._flat(atoms_2))
    cont_differ = len(c1) != len(c2) or any(
        abs(u - v) > 1e-12 for u, v in zip(c1, c2))
    atoms_differ = a1 != a2
    singular = cont_differ or atoms_differ
    res = RichResult(payload={"estimate": 1.0 if singular else 0.0,
                              "mutually_singular": singular,
                              "continuous_parts_differ": cont_differ,
                              "atomic_supports_differ": atoms_differ,
                              "method": "DP mutual singularity (GvdV 2017 Thm 4.21)"})
    return with_describe_pointer(res, "gh_c4_15")


def cheatsheet():
    return "gh_c4_15: Mutual singularity of Dirichlet processes"
