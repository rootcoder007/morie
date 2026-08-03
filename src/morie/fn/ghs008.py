# morie.fn -- function file (rootcoder007/morie)
"""Prior weights by normalization.

Implements eq. (3.1), p.29 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_normalized_weights_prior"]


def ghosal_ch3_normalized_weights_prior(Y_j, k=None):
    """p_k = Y_k / sum_j Y_j (eq. 3.1). Keys: distribution."""
    p = _bnp.normalize_weights(Y_j)
    est = p[int(k)] if k is not None else p[0]
    res = RichResult(payload={"estimate": est, "distribution": p,
                              "method": "normalized weights (GvdV 2017 eq. 3.1)"})
    return with_describe_pointer(res, "ghs008")


def cheatsheet():
    return "ghs008: Prior weights by normalization"
