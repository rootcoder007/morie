# morie.fn -- function file (rootcoder007/morie)
"""NTR Lévy (Laplace) functional.

Implements sec. 13.4 (Laplace functional form) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ntr_levy"]


def ghosal_ntr_levy(f_vals, nu_masses):
    """E exp(-int f dM) = exp(-int (1 - e^{-f u}) dnu) for a
    discrete Levy measure nu = sum m_j delta_{u_j} paired with f
    values (sec. 13.4): computed exactly. Keys: estimate."""
    fs = _bnp._flat(f_vals)
    ms = _bnp._flat(nu_masses)
    exponent = sum(m * (1.0 - math.exp(-f)) for f, m in zip(fs, ms))
    val = math.exp(-exponent)
    res = RichResult(payload={"estimate": val,
                              "exponent": exponent,
                              "method": "NTR Laplace functional (GvdV 2017 sec. 13.4)"})
    return with_describe_pointer(res, "gh_c13_9")


def cheatsheet():
    return "gh_c13_9: NTR Lévy (Laplace) functional"
