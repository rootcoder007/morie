# morie.fn -- function file (rootcoder007/morie)
"""Hellinger distance.

Implements Appendix B of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_hellinger_dist"]


def ghosal_hellinger_dist(p, q):
    """d_H^2(P,Q) = 1 - int sqrt(p q) = 1 - rho_{1/2} (App B); also
    d_H^2 <= d_TV <= d_H sqrt(2 - d_H^2). Keys: estimate."""
    p = _bnp.normalize_weights(p)
    q = _bnp.normalize_weights(q)
    rho = sum(math.sqrt(a * b) for a, b in zip(p, q))
    h2 = 1.0 - rho
    tv = 0.5 * sum(abs(a - b) for a, b in zip(p, q))
    h = math.sqrt(max(h2, 0.0))
    res = RichResult(payload={"estimate": h2,
                              "affinity": rho,
                              "inequalities_hold":
                                  h2 <= tv + 1e-12 and
                                  tv <= h * math.sqrt(2.0 - h2)
                                  + 1e-12,
                              "method": "Hellinger distance (GvdV 2017 App B)"})
    return with_describe_pointer(res, "gh_ap_a4")


def cheatsheet():
    return "gh_ap_a4: Hellinger distance"
