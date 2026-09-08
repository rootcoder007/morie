# morie.fn -- function file (rootcoder007/morie)
"""DP posterior (conjugacy).

Implements Theorem 4.6, eq. (4.10)-(4.12) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_post"]


def ghosal_dp_post(G0_A, alpha, n_in_A, n):
    """G | X ~ DP(alpha + sum delta_Xi) (Thm 4.6). Posterior mean
    (4.11): E(P(A)|X) = M/(M+n) G0(A) + n/(M+n) P_n(A); posterior
    variance (4.12) <= 1/(4(1+M+n)). Keys: estimate."""
    g = float(_bnp._flat(G0_A)[0])
    M = float(alpha)
    n = float(n)
    pn = float(n_in_A) / n if n > 0 else 0.0
    mean = M / (M + n) * g + n / (M + n) * pn
    var = mean * (1.0 - mean) / (1.0 + M + n)
    res = RichResult(payload={"estimate": mean,
                              "posterior_var": var,
                              "var_bound": 1.0 / (4.0 * (1.0 + M + n)),
                              "posterior_precision": M + n,
                              "method": "DP conjugate posterior (GvdV 2017 eq. 4.10-4.12)"})
    return with_describe_pointer(res, "gh_c4_6")


def cheatsheet():
    return "gh_c4_6: DP posterior (conjugacy)"


# compact alias per ledger/NAMING.md
ghosaldppost = ghosal_dp_post
