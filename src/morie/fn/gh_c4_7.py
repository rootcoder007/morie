# morie.fn -- function file (rootcoder007/morie)
"""DP predictive distribution.

Implements eq. (4.13) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_pred"]


def ghosal_dp_pred(x_seen, alpha, x_new_equals=None):
    """X_{n+1} | X_1..X_n ~ (alpha G0-bar + sum delta_Xi)/(|alpha|+n)
    (eq. 4.13): each past point gets weight 1/(M+n), a fresh draw
    from G0-bar gets M/(M+n). Keys: estimate."""
    xs = _bnp._flat(x_seen)
    M = float(alpha)
    n = len(xs)
    w_new = M / (M + n)
    w_each = 1.0 / (M + n)
    if x_new_equals is not None:
        t = float(x_new_equals)
        est = sum(w_each for v in xs if v == t)
    else:
        est = w_new
    res = RichResult(payload={"estimate": est,
                              "weight_fresh": w_new,
                              "weight_per_obs": w_each,
                              "method": "generalized Polya urn predictive (GvdV 2017 eq. 4.13)"})
    return with_describe_pointer(res, "gh_c4_7")


def cheatsheet():
    return "gh_c4_7: DP predictive distribution"


# compact alias per ledger/NAMING.md
ghosaldppred = ghosal_dp_pred
