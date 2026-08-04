# morie.fn -- function file (rootcoder007/morie)
"""Ewens sampling formula.

Implements sec. 14.1 (Proposition 4.10) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ewens_esf"]


def ghosal_ewens_esf(multiplicities, alpha=1.0):
    """P(multiplicity class C(m_1..m_n)) = n!/alpha^{[n]}
    prod alpha^{m_i} / (i^{m_i} m_i!) (Prop 4.10). ``multiplicities``
    maps size i -> m_i (list, index 0 = size 1). Keys: estimate."""
    ms = [int(v) for v in _bnp._flat(multiplicities)]
    n = sum((i + 1) * m for i, m in enumerate(ms))
    log_asc = sum(math.log(alpha + i) for i in range(n))
    lp = math.lgamma(n + 1.0) - log_asc
    for i, m in enumerate(ms):
        size = i + 1
        lp += m * math.log(alpha) - m * math.log(size) \
            - math.lgamma(m + 1.0)
    res = RichResult(payload={"estimate": math.exp(lp),
                              "log_prob": lp, "n": n,
                              "method": "Ewens sampling formula (GvdV 2017 Prop 4.10)"})
    return with_describe_pointer(res, "gh_c14_2")


def cheatsheet():
    return "gh_c14_2: Ewens sampling formula"


# compact alias per ledger/NAMING.md
ghosalewensesf = ghosal_ewens_esf
