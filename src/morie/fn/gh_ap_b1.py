# morie.fn -- function file (rootcoder007/morie)
"""KL divergence properties (Pinsker).

Implements Appendix B of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_kl_props"]


def ghosal_kl_props(p, q):
    """KL >= 0 with equality iff P = Q, and d_TV^2 <= KL/2 (Pinsker)
    (App B). All three checked. Keys: estimate."""
    p = _bnp.normalize_weights(p)
    q = _bnp.normalize_weights(q)
    kl = sum(a * math.log(a / max(b, 1e-300))
             for a, b in zip(p, q) if a > 0)
    tv = 0.5 * sum(abs(a - b) for a, b in zip(p, q))
    res = RichResult(payload={"estimate": kl,
                              "nonneg": kl >= -1e-15,
                              "pinsker_holds": tv * tv <= kl / 2.0
                              + 1e-12,
                              "zero_iff_equal": (kl < 1e-14)
                              == (tv < 1e-14),
                              "method": "KL properties + Pinsker (GvdV 2017 App B)"})
    return with_describe_pointer(res, "gh_ap_b1")


def cheatsheet():
    return "gh_ap_b1: KL divergence properties (Pinsker)"


# compact alias per ledger/NAMING.md
ghosalklprops = ghosal_kl_props
