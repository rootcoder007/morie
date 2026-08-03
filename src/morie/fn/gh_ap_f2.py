# morie.fn -- function file (rootcoder007/morie)
"""Glivenko-Cantelli theorem.

Implements Appendix F of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_glivenko"]


def ghosal_glivenko(ns=(100, 1000, 10000), seed=42):
    """||F_n - F||_infty -> 0 a.s. for any F (App F): empirical
    KS distance for a uniform truth shrinks along n. Keys: estimate."""
    rng = np.random.default_rng(seed)
    sups = []
    for n in ns:
        data = sorted(float(rng.uniform(0, 1)) for _ in range(n))
        sup = 0.0
        for i, v in enumerate(data):
            sup = max(sup, abs((i + 1) / n - v), abs(i / n - v))
        sups.append(sup)
    res = RichResult(payload={"estimate": sups[-1],
                              "sup_by_n": sups,
                              "vanishing": sups[-1] < sups[0]
                              and sups[-1] < 0.02,
                              "method": "Glivenko-Cantelli (GvdV 2017 App F)"})
    return with_describe_pointer(res, "gh_ap_f2")


def cheatsheet():
    return "gh_ap_f2: Glivenko-Cantelli theorem"
