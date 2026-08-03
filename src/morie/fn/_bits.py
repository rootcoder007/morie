# morie.fn -- function file (rootcoder007/morie)
"""Finite-level density p_m.

Implements eq. (3.19), p.44 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["_bits"]


def _bits(x, cell_masses, depth):
    """p_m = sum_{e in E^m} (P(A_e)/mu(A_e)) 1_{A_e} (eq. 3.19)
    for canonical dyadic cells mu(A_e) = 2^-m: evaluates p_m at x.
    Keys: distribution."""
    masses = _bnp._flat(cell_masses)
    m = int(depth)
    if len(masses) != 2 ** m:
        raise ValueError("need 2^depth cell masses")
    bits = _bnp._bits(_bnp._flat(x)[0], m)
    idx = 0
    for b in bits:
        idx = 2 * idx + b
    dens = masses[idx] * (2.0 ** m)
    res = RichResult(payload={"estimate": dens, "distribution": dens,
                              "cell_index": idx,
                              "total_mass": sum(masses),
                              "method": "finite-level density (GvdV 2017 eq. 3.19)"})
    return with_describe_pointer(res, "ghs026")


def cheatsheet():
    return "ghs026: Finite-level density p_m"
