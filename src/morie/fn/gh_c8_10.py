# morie.fn -- function file (rootcoder007/morie)
"""White-noise model contraction.

Implements Example 8.6, eq. (8.1) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_wn_crt"]


def ghosal_wn_crt(s_true=1.0, alpha_prior=1.0, ns=(100, 10000)):
    """theta_i | X ~ N(n X_i/(n + i^{2a+1}), 1/(n + i^{2a+1}))
    (eq. 8.1): expected posterior L2 risk sums bias^2 + two variance
    series; rate n^{-2 min(a,s)/(2a+1)} (Ex 8.6). Keys: estimate."""
    a = float(alpha_prior)
    risks = []
    for n in ns:
        tot = 0.0
        for i in range(1, 4000):
            th0 = float(i) ** (-(s_true + 0.5))
            lam = float(i) ** (2.0 * a + 1.0)
            tot += (th0 * lam / (n + lam)) ** 2 \
                + n / (n + lam) ** 2 + 1.0 / (n + lam)
        risks.append(tot)
    rate_hat = math.log(risks[0] / risks[-1]) \
        / math.log(float(ns[-1]) / ns[0])
    expect = 2.0 * min(a, s_true) / (2.0 * a + 1.0)
    res = RichResult(payload={"estimate": rate_hat,
                              "expected_exponent": expect,
                              "risk_by_n": risks,
                              "method": "white-noise contraction (GvdV 2017 Ex 8.6, eq. 8.1)"})
    return with_describe_pointer(res, "gh_c8_10")


def cheatsheet():
    return "gh_c8_10: White-noise model contraction"


# compact alias per ledger/NAMING.md
ghosalwncrt = ghosal_wn_crt
