# morie.fn -- function file (rootcoder007/morie)
"""Countable Dirichlet posterior, k cells.

Implements eq. (3.6), p.32 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_countable_dirichlet_posterior_k"]


def ghosal_ch3_countable_dirichlet_posterior_k(alpha, counts, k,
                                               alpha_tail=0.0):
    """Dir(k+1; alpha_j + N_j, tail alpha + n - sum_{j<=k} N_j)
    (eq. 3.6). Keys: posterior."""
    a = _bnp._flat(alpha)[:int(k)]
    N = _bnp._flat(counts)
    n = sum(N)
    upd = [ai + Ni for ai, Ni in zip(a, N[:int(k)])]
    tail = float(alpha_tail) + n - sum(N[:int(k)])
    res = RichResult(payload={"estimate": upd[0],
                              "posterior": upd + [tail],
                              "method": "countable Dirichlet posterior (GvdV 2017 eq. 3.6)"})
    return with_describe_pointer(res, "ghs013")


def cheatsheet():
    return "ghs013: Countable Dirichlet posterior, k cells"
