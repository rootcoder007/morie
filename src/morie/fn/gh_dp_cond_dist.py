# morie.fn -- function file (rootcoder007/morie)
"""DP conditional distribution on a complement.

Implements Theorem 4.5 (localization) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_conditional_distribution"]


def ghosal_dp_conditional_distribution(base_masses_in_Ac, w=0.5,
                                       seed=42):
    """Given P(A) = w, the conditional measure on A^c is
    P_{A^c} ~ DP(alpha|_{A^c}), independent of w (Theorem 4.5). The
    returned Dirichlet parameters are just the restricted base
    masses. Keys: estimate."""
    a = _bnp._flat(base_masses_in_Ac)
    rng = np.random.default_rng(seed)
    p = _bnp.normalize_weights(
        [float(rng.gamma(max(ai, 1e-12), 1.0)) for ai in a])
    res = RichResult(payload={"estimate": p[0],
                              "dir_params": a, "P_cond": p,
                              "independent_of_w": True,
                              "method": "DP localized conditional (GvdV 2017 Thm 4.5)"})
    return with_describe_pointer(res, "gh_dp_cond_dist")


def cheatsheet():
    return "gh_dp_cond_dist: DP conditional distribution on a complement"
