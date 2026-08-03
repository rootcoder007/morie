# morie.fn -- function file (rootcoder007/morie)
"""Mixture of Dirichlet processes.

Implements eq. (4.27) + prior moments p.87 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_mix_dp"]


def ghosal_mix_dp(G0_A_by_xi, alpha_by_xi, pi_weights):
    """MDP: xi ~ pi, P | xi ~ DP(alpha_xi). Prior mean
    E P(A) = int G0_xi(A) dpi(xi); prior variance adds the
    within-DP part int G0(1-G0)/(1+|alpha_xi|) dpi and the
    between-component spread (GvdV 2017 sec. 4.5, p.87).
    Keys: estimate."""
    gs = _bnp._flat(G0_A_by_xi)
    Ms = _bnp._flat(alpha_by_xi)
    w = _bnp.normalize_weights(pi_weights)
    mean = sum(wi * g for wi, g in zip(w, gs))
    within = sum(wi * g * (1.0 - g) / (1.0 + m)
                 for wi, g, m in zip(w, gs, Ms))
    between = sum(wi * (g - mean) ** 2 for wi, g in zip(w, gs))
    res = RichResult(payload={"estimate": mean,
                              "variance": within + between,
                              "var_within": within,
                              "var_between": between,
                              "method": "MDP prior moments (GvdV 2017 sec. 4.5)"})
    return with_describe_pointer(res, "gh_c4_20")


def cheatsheet():
    return "gh_c4_20: Mixture of Dirichlet processes"
