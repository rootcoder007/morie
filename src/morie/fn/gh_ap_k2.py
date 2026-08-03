# morie.fn -- function file (rootcoder007/morie)
"""Assouad's lemma.

Implements Appendix K of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_assouad_lemma"]


def ghosal_assouad_lemma(m=8, per_coord_sep=0.1, affinity=0.8):
    """R_n >= (m/2) * sep * min affinity over one-coordinate flips
    (App K, Assouad): lower bound on the minimax risk from an
    m-dimensional hypercube of hypotheses. Keys: estimate."""
    bound = 0.5 * m * per_coord_sep * affinity
    res = RichResult(payload={"estimate": bound,
                              "grows_with_m": True,
                              "method": "Assouad lower bound (GvdV 2017 App K)"})
    return with_describe_pointer(res, "gh_ap_k2")


def cheatsheet():
    return "gh_ap_k2: Assouad's lemma"
