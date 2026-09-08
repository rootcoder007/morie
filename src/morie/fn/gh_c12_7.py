# morie.fn -- function file (rootcoder007/morie)
"""Strict semiparametric BvM conditions.

Implements sec. 12.3.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_strict_sbvm"]


def ghosal_strict_sbvm(prior_mass_ok=True, lan_remainder=0.01,
                       change_of_measure_gap=0.02, tol=0.05):
    """The exact semiparametric BvM needs (i) prior mass around the
    least-favorable direction, (ii) LAN remainder -> 0, (iii) a
    change-of-measure (prior invariance) condition (sec. 12.3.2).
    Aggregates the three checks. Keys: estimate."""
    ok = bool(prior_mass_ok) and float(lan_remainder) < tol \
        and float(change_of_measure_gap) < tol
    score = (0.0 if not prior_mass_ok else 1.0) \
        - float(lan_remainder) - float(change_of_measure_gap)
    res = RichResult(payload={"estimate": score,
                              "bvm_holds": ok,
                              "conditions": [bool(prior_mass_ok),
                                             float(lan_remainder)
                                             < tol,
                                             float(
                                                 change_of_measure_gap)
                                             < tol],
                              "method": "strict semiparametric BvM (GvdV 2017 sec. 12.3.2)"})
    return with_describe_pointer(res, "gh_c12_7")


def cheatsheet():
    return "gh_c12_7: Strict semiparametric BvM conditions"
