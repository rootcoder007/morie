# morie.fn -- function file (rootcoder007/morie)
"""Absolute-continuity condition.

Implements Theorem 3.16, eq. (3.16), p.44 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_tailfree_abs_continuity_cond"]


def ghosal_ch3_tailfree_abs_continuity_cond(EV2_by_level, mu_exponent=2.0,
                                            m_max=None):
    """sup_m max_e E(prod_j V^2) / mu^2(A_e) < infty (eq. 3.16). For
    canonical partitions mu(A_e) = 2^-m so the ratio at level m is
    prod_{j<=m} 4 E(V_j^2); the sup over m is reported. Keys: value."""
    ev2 = _bnp._flat(EV2_by_level)
    m_max = len(ev2) if m_max is None else int(m_max)
    ratios = []
    prod = 1.0
    for j in range(m_max):
        prod *= float(mu_exponent) ** 2 * ev2[j]
        ratios.append(prod)
    sup = max(ratios)
    res = RichResult(payload={"estimate": sup, "value": sup,
                              "ratios_by_level": ratios,
                              "finite": math.isfinite(sup),
                              "method": "abs continuity condition (GvdV 2017 eq. 3.16)"})
    return with_describe_pointer(res, "ghs023")


def cheatsheet():
    return "ghs023: Absolute-continuity condition"
