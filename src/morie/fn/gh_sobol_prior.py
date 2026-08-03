# morie.fn -- function file (rootcoder007/morie)
"""Sobolev sequence prior.

Implements sec. 9.5.4 (theta_j ~ N(0, j^{-2s-1})) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_sobolev_prior"]


def ghosal_sobolev_prior(smoothness=1.0, n_terms=2000, seed=42):
    """theta_j ~ N(0, j^{-2s-1}): E sum j^{2t} theta_j^2 =
    sum j^{2t - 2s - 1} is finite iff t < s -- draws sit in every
    Sobolev space of order t < s, giving rate n^{-2s/(2s+1)}
    (sec. 9.5.4). Keys: estimate."""
    s = float(smoothness)
    rng = np.random.default_rng(seed)
    th = [float(rng.normal(0, 1)) * float(j) ** (-(s + 0.5))
          for j in range(1, n_terms + 1)]
    norm_below = sum(float(j) ** (2.0 * (s - 0.25)) * th[j - 1] ** 2
                     for j in range(1, n_terms + 1))
    # expected norm at t = s diverges logarithmically:
    e_norm_at_s = sum(1.0 / j for j in range(1, n_terms + 1))
    res = RichResult(payload={"estimate": norm_below,
                              "finite_below_s":
                                  math.isfinite(norm_below),
                              "divergent_at_s_partial": e_norm_at_s,
                              "rate": "n^(-2s/(2s+1))",
                              "method": "Sobolev sequence prior (GvdV 2017 sec. 9.5.4)"})
    return with_describe_pointer(res, "gh_sobol_prior")


def cheatsheet():
    return "gh_sobol_prior: Sobolev sequence prior"
