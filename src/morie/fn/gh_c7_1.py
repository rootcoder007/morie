# morie.fn -- function file (rootcoder007/morie)
"""Pólya tree KL property.

Implements sec. 7.1.1 (canonical PT with summable 1/a_m) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_pt_kl_prop"]


def ghosal_pt_kl_prop(a_exponent=2.0, m_max=200):
    """PT*(lambda, a_m) with a_m = m^2 (more generally
    sum a_m^{-1} < infty plus growth) has the KL property at smooth
    densities (sec. 7.1.1; absolute continuity via Thm 3.16/3.22).
    Checks the two canonical series: sum 1/a_m and the eq. (3.17)
    variance series sum 1/(4(2 a_m + 1)). Keys: estimate."""
    s_inv = sum(1.0 / float(m) ** a_exponent
                for m in range(1, m_max + 1))
    s_var = sum(1.0 / (4.0 * (2.0 * float(m) ** a_exponent + 1.0))
                for m in range(1, m_max + 1))
    summable = a_exponent > 1.0
    res = RichResult(payload={"estimate": s_inv,
                              "variance_series": s_var,
                              "kl_property": summable,
                              "method": "PT KL-property series (GvdV 2017 sec. 7.1.1)"})
    return with_describe_pointer(res, "gh_c7_1")


def cheatsheet():
    return "gh_c7_1: Pólya tree KL property"


# compact alias per ledger/NAMING.md
ghosalptklprop = ghosal_pt_kl_prop
