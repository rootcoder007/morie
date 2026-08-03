# morie.fn -- function file (rootcoder007/morie)
"""Random basis expansion.

Implements sec. 2.2 (random series) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch2_random_basis_expansion"]


def ghosal_ch2_random_basis_expansion(beta_j, psi_j=None, J=None,
                                      x=0.3):
    """f = sum_{j <= J} beta_j psi_j (sec. 2.2): evaluates the random
    series at x on the cosine basis when psi_j is not supplied.
    Keys: distribution."""
    b = _bnp._flat(beta_j)
    if J is not None:
        b = b[:int(J)]
    if psi_j is None:
        vals = [math.sqrt(2.0) * math.cos((j + 1) * math.pi * x)
                for j in range(len(b))]
    else:
        vals = _bnp._flat(psi_j)[:len(b)]
    f = sum(bj * pj for bj, pj in zip(b, vals))
    res = RichResult(payload={"estimate": f, "distribution": f,
                              "J": len(b),
                              "method": "random basis expansion (GvdV 2017 sec. 2.2)"})
    return with_describe_pointer(res, "ghs002")


def cheatsheet():
    return "ghs002: Random basis expansion"
