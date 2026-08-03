# morie.fn -- function file (rootcoder007/morie)
"""Besov spike-and-slab wavelet prior.

Implements sec. 10.3.2 (level-indexed variant) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_besov_prior"]


def ghosal_besov_prior(s=1.0, J=8, pi_j=0.5, seed=42):
    """theta_{jk} ~ pi_j N(0, 2^{-j(2s+1)}) + (1 - pi_j) delta_0
    (sec. 10.3.2): draws a wavelet coefficient array and checks the
    Besov-type norm sum_j 2^{j(2s'+1)} sum_k theta_jk^2 is finite for
    s' < s. Keys: estimate."""
    rng = np.random.default_rng(seed)
    norm_below = 0.0
    n_active = 0
    for j in range(J):
        sd = 2.0 ** (-j * (2.0 * s + 1.0) / 2.0)
        lvl = 0.0
        for k in range(2 ** j):
            if float(rng.uniform(0, 1)) < pi_j:
                th = sd * float(rng.normal(0, 1))
                lvl += th * th
                n_active += 1
        norm_below += 2.0 ** (j * (2.0 * (s - 0.25) + 1.0)) * lvl
    res = RichResult(payload={"estimate": norm_below,
                              "n_active": n_active,
                              "finite": math.isfinite(norm_below),
                              "method": "Besov spike-slab prior (GvdV 2017 sec. 10.3.2)"})
    return with_describe_pointer(res, "gh_besov_prior")


def cheatsheet():
    return "gh_besov_prior: Besov spike-and-slab wavelet prior"
