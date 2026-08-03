# morie.fn -- function file (rootcoder007/morie)
"""Poisson-Kingman process.

Implements sec. 14.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_pk_process"]


def ghosal_pk_process(n_jumps=500, seed=42):
    """G = sum_k (J_k / T) delta_{theta_k}, J_k the jumps of a
    Poisson process with Levy measure rho, T = sum J_k (sec. 14.5):
    gamma-Levy jumps normalize to a Dirichlet-type measure.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    J = [float(rng.gamma(1.0 / n_jumps * 3.0, 1.0))
         for _ in range(n_jumps)]
    T = sum(J)
    W = [j / T for j in J]
    res = RichResult(payload={"estimate": max(W),
                              "total_mass": sum(W),
                              "method": "Poisson-Kingman process (GvdV 2017 sec. 14.5)"})
    return with_describe_pointer(res, "gh_c14_12")


def cheatsheet():
    return "gh_c14_12: Poisson-Kingman process"
