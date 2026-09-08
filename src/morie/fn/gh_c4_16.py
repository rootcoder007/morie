# morie.fn -- function file (rootcoder007/morie)
"""DP tail bounds.

Implements Theorem 4.22, eq. (4.24) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_tails"]


def ghosal_dp_tails(MG_x, r=2.0):
    """exp(-r log|log MG(x)| / MG(x)) <= F(x) <=
    exp(-1/(MG(x) |log MG(x)|^r)) a.s. eventually, r > 1 (eq. 4.24):
    the DP tail is exponentially thinner than the base measure tail.
    Keys: estimate."""
    m = float(_bnp._flat(MG_x)[0])
    if not 0.0 < m < 1.0:
        raise ValueError("MG(x) must lie in (0,1) for the tail zone")
    r = float(r)
    ll = abs(math.log(m))
    lo = math.exp(-r * math.log(ll) / m) if ll > 1.0 else 0.0
    hi = math.exp(-1.0 / (m * ll ** r))
    res = RichResult(payload={"estimate": hi, "lower": lo,
                              "upper": hi,
                              "thinner_than_base": hi < m,
                              "method": "DP tail bounds (GvdV 2017 eq. 4.24)"})
    return with_describe_pointer(res, "gh_c4_16")


def cheatsheet():
    return "gh_c4_16: DP tail bounds"


# compact alias per ledger/NAMING.md
ghosaldptails = ghosal_dp_tails
