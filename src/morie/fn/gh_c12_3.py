# morie.fn -- function file (rootcoder007/morie)
"""Strong approximation for the DP.

Implements sec. 12.2.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_strong_apx_dp"]


def ghosal_strong_apx_dp(n=3000, seed=42):
    """sup_t |sqrt(n)(F_n(t) - F0(t)) - B(F0(t))| -> 0 a.s.
    (sec. 12.2.1): the empirical bridge stays uniformly bounded at
    the Kolmogorov-Smirnov scale (sup-statistic has the KS law).
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    data = sorted(float(rng.uniform(0, 1)) for _ in range(n))
    sup = 0.0
    for i, v in enumerate(data):
        sup = max(sup, abs((i + 1) / n - v), abs(i / n - v))
    ks = math.sqrt(n) * sup
    res = RichResult(payload={"estimate": ks,
                              "typical_ks_range": ks < 3.0,
                              "method": "strong approximation (GvdV 2017 sec. 12.2.1)"})
    return with_describe_pointer(res, "gh_c12_3")


def cheatsheet():
    return "gh_c12_3: Strong approximation for the DP"
