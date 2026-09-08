# morie.fn -- function file (rootcoder007/morie)
"""Neutral-to-the-right processes.

Implements sec. 13.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ntr_def"]


def ghosal_ntr_def(increments, seed=42):
    """F(t) = 1 - exp(-M(t)) for M an independent-increment process
    (sec. 13.4): NTR means the relative survival fractions over
    disjoint intervals are independent. Builds F from supplied
    positive increments of M. Keys: estimate."""
    inc = _bnp._flat(increments)
    if any(v < 0 for v in inc):
        raise ValueError("increments must be nonnegative")
    M = 0.0
    F = []
    for v in inc:
        M += v
        F.append(1.0 - math.exp(-M))
    res = RichResult(payload={"estimate": F[-1],
                              "F_path": F,
                              "nondecreasing": all(
                                  F[i + 1] >= F[i] - 1e-15
                                  for i in range(len(F) - 1)),
                              "method": "NTR construction (GvdV 2017 sec. 13.4)"})
    return with_describe_pointer(res, "gh_c13_8")


def cheatsheet():
    return "gh_c13_8: Neutral-to-the-right processes"


# compact alias per ledger/NAMING.md
ghosalntrdef = ghosal_ntr_def
