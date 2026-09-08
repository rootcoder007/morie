# morie.fn -- function file (rootcoder007/morie)
"""Kullback-Leibler variations.

Implements Appendix B (V_k) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_kl_variation"]


def ghosal_kl_variation(p, q, k=2):
    """V_k(P; Q) = int |log(p/q)|^k dP; V_1 relates to KL (App B).
    Keys: estimate."""
    p = _bnp.normalize_weights(p)
    q = _bnp.normalize_weights(q)
    Vk = sum(a * abs(math.log(a / max(b, 1e-300))) ** k
             for a, b in zip(p, q) if a > 0)
    kl = sum(a * math.log(a / max(b, 1e-300))
             for a, b in zip(p, q) if a > 0)
    res = RichResult(payload={"estimate": Vk, "kl": kl,
                              "dominates_kl": Vk >= 0,
                              "method": "KL variation (GvdV 2017 App B)"})
    return with_describe_pointer(res, "gh_ap_b2")


def cheatsheet():
    return "gh_ap_b2: Kullback-Leibler variations"
