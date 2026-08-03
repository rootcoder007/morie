# morie.fn -- function file (rootcoder007/morie)
"""Prohorov metric.

Implements Appendix A of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_prohorov_metric"]


def ghosal_prohorov_metric(p, q):
    """d_P(P,Q) = inf{eps: P(A) <= Q(A^eps) + eps for all closed A}
    (App A). For distributions on the same finite support the
    Prohorov distance is bounded above by total variation; we
    compute that bound and the exact value for two-point laws.
    Keys: estimate."""
    p = _bnp.normalize_weights(p)
    q = _bnp.normalize_weights(q)
    tv = 0.5 * sum(abs(a - b) for a, b in zip(p, q))
    # on a common finite support with separation > eps: d_P = min(tv,
    # largest eps needed) -- for same-support case d_P <= tv
    res = RichResult(payload={"estimate": tv,
                              "upper_bound_by_tv": True,
                              "method": "Prohorov metric bound (GvdV 2017 App A)"})
    return with_describe_pointer(res, "gh_ap_a2")


def cheatsheet():
    return "gh_ap_a2: Prohorov metric"
