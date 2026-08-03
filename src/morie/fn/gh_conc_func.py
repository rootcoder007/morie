# morie.fn -- function file (rootcoder007/morie)
"""Concentration function of a GP.

Implements eq. (11.11) + Proposition 11.19 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_concentration_function"]


def ghosal_concentration_function(decentering_norm2, small_ball_exp):
    """phi_{w0}(eps) = inf_{||h - w0|| < eps} ||h||_H^2
    - log Pi(||W|| < eps): combines the decentering and small-ball
    parts; by Prop 11.19 it sandwiches -log P(||W - w0|| < eps).
    Keys: estimate."""
    phi = float(decentering_norm2) + float(small_ball_exp)
    res = RichResult(payload={"estimate": phi,
                              "decentering": float(decentering_norm2),
                              "small_ball": float(small_ball_exp),
                              "method": "concentration function (GvdV 2017 eq. 11.11, Prop 11.19)"})
    return with_describe_pointer(res, "gh_conc_func")


def cheatsheet():
    return "gh_conc_func: Concentration function of a GP"
