# morie.fn -- function file (rootcoder007/morie)
"""Le Cam's inequality.

Implements Lemma 6.46 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_lecam_consist"]


def ghosal_lecam_consist(tv_P0_PU, P0_phi, Pi_U, int_V_P_1mphi):
    """P0 Pi(V | X) <= d_TV(P0, P_U) + P0 phi +
    (1/Pi(U)) int_V P(1-phi) dPi (Lemma 6.46): evaluates the bound
    from its three ingredients. Keys: estimate."""
    bound = float(tv_P0_PU) + float(P0_phi) \
        + float(int_V_P_1mphi) / float(Pi_U)
    res = RichResult(payload={"estimate": bound,
                              "terms": [float(tv_P0_PU),
                                        float(P0_phi),
                                        float(int_V_P_1mphi)
                                        / float(Pi_U)],
                              "method": "Le Cam inequality (GvdV 2017 Lemma 6.46)"})
    return with_describe_pointer(res, "gh_c6_13")


def cheatsheet():
    return "gh_c6_13: Le Cam's inequality"
