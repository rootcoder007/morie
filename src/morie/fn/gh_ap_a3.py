# morie.fn -- function file (rootcoder007/morie)
"""Total-variation distance.

Implements Appendix A/B of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_tv_distance"]


def ghosal_tv_distance(p, q):
    """d_TV(P,Q) = sup_A |P(A) - Q(A)| = (1/2) ||p - q||_1 (App B).
    Both forms computed; they agree exactly. Keys: estimate."""
    p = _bnp.normalize_weights(p)
    q = _bnp.normalize_weights(q)
    half_l1 = 0.5 * sum(abs(a - b) for a, b in zip(p, q))
    sup_A = sum(max(a - b, 0.0) for a, b in zip(p, q))
    res = RichResult(payload={"estimate": half_l1,
                              "sup_form": sup_A,
                              "forms_agree": abs(half_l1 - sup_A)
                              < 1e-12,
                              "method": "total variation (GvdV 2017 App B)"})
    return with_describe_pointer(res, "gh_ap_a3")


def cheatsheet():
    return "gh_ap_a3: Total-variation distance"
