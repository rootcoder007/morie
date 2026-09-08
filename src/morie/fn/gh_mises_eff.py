# morie.fn -- function file (rootcoder007/morie)
"""Mises differentiability and efficiency.

Implements sec. 12.3 (von Mises expansion) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_mises_efficiency"]


def ghosal_mises_efficiency(t_eval=0.5, h=0.02):
    """psi(P) - psi(P0) = int psi-tilde d(P - P0) + o(||P - P0||):
    the influence function of psi(P) = P(X <= t) is
    psi-tilde(x) = 1{x <= t} - P0(X <= t) (sec. 12.3). Verifies the
    first-order expansion exactly for a contamination path
    P_h = (1-h) P0 + h delta_x. Keys: estimate."""
    P0_t = t_eval                     # uniform truth
    x_pts = (0.2, 0.8)
    gaps = []
    for x in x_pts:
        psi_h = (1.0 - h) * P0_t + h * (1.0 if x <= t_eval else 0.0)
        infl = (1.0 if x <= t_eval else 0.0) - P0_t
        first_order = P0_t + h * infl
        gaps.append(abs(psi_h - first_order))
    res = RichResult(payload={"estimate": max(gaps),
                              "expansion_exact": max(gaps) < 1e-14,
                              "influence_at_02": (1.0 if 0.2
                                                  <= t_eval else 0.0)
                              - P0_t,
                              "method": "von Mises expansion (GvdV 2017 sec. 12.3)"})
    return with_describe_pointer(res, "gh_mises_eff")


def cheatsheet():
    return "gh_mises_eff: Mises differentiability and efficiency"
