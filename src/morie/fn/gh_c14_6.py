# morie.fn -- function file (rootcoder007/morie)
"""Species-sampling posterior.

Implements sec. 14.2.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ssp_post"]


def ghosal_ssp_post(counts, alpha=2.0):
    """After observing species counts the predictive puts weight
    (n_k adjusted) on seen species and the EPPF-derived weight on a
    new one -- for the DP-SSP: n_k/(alpha+n) and alpha/(alpha+n)
    (sec. 14.2.1). Keys: estimate."""
    ns = [float(v) for v in _bnp._flat(counts)]
    n = sum(ns)
    seen = [v / (alpha + n) for v in ns]
    new = alpha / (alpha + n)
    res = RichResult(payload={"estimate": new,
                              "seen_weights": seen,
                              "total": sum(seen) + new,
                              "method": "SSP posterior predictive (GvdV 2017 sec. 14.2.1)"})
    return with_describe_pointer(res, "gh_c14_6")


def cheatsheet():
    return "gh_c14_6: Species-sampling posterior"


# compact alias per ledger/NAMING.md
ghosalssppost = ghosal_ssp_post
