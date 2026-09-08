# morie.fn -- function file (rootcoder007/morie)
"""Rényi divergence.

Implements Appendix B of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_renyi_div"]


def ghosal_renyi_div(p, q, alpha=0.5):
    """D_alpha(P||Q) = (alpha - 1)^{-1} log int p^alpha q^{1-alpha}
    for alpha in (0,1) (App B); D_{1/2} = -2 log(1 - d_H^2).
    Keys: estimate."""
    p = _bnp.normalize_weights(p)
    q = _bnp.normalize_weights(q)
    rho = sum(a ** alpha * b ** (1.0 - alpha) for a, b in zip(p, q))
    D = math.log(rho) / (alpha - 1.0)
    h2 = 1.0 - sum(math.sqrt(a * b) for a, b in zip(p, q))
    res = RichResult(payload={"estimate": D,
                              "hellinger_link_gap":
                                  abs(D + 2.0 * math.log(1.0 - h2))
                                  if abs(alpha - 0.5) < 1e-12
                                  else None,
                              "method": "Renyi divergence (GvdV 2017 App B)"})
    return with_describe_pointer(res, "gh_ap_b3")


def cheatsheet():
    return "gh_ap_b3: Rényi divergence"


# compact alias per ledger/NAMING.md
ghosalrenyidiv = ghosal_renyi_div
