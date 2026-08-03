# morie.fn -- function file (rootcoder007/morie)
"""Countable Dirichlet posterior, l cells.

Implements eq. (3.5), p.32 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_countable_dirichlet_posterior_l"]


def ghosal_ch3_countable_dirichlet_posterior_l(alpha, counts, l,
                                               alpha_tail=0.0):
    """Dir(l+1; alpha_1+N_1, ..., alpha_l+N_l, tail + n - sum N)
    (eq. 3.5). Keys: posterior."""
    a = _bnp._flat(alpha)[:int(l)]
    N = _bnp._flat(counts)
    n = sum(N)
    upd = [ai + Ni for ai, Ni in zip(a, N[:int(l)])]
    tail = float(alpha_tail) + n - sum(N[:int(l)])
    res = RichResult(payload={"estimate": upd[0],
                              "posterior": upd + [tail],
                              "method": "countable Dirichlet posterior (GvdV 2017 eq. 3.5)"})
    return with_describe_pointer(res, "ghs012")


def cheatsheet():
    return "ghs012: Countable Dirichlet posterior, l cells"
