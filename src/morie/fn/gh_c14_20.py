# morie.fn -- function file (rootcoder007/morie)
"""Probit stick-breaking process.

Implements sec. 14.9.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_probit_sbp"]


def ghosal_probit_sbp(x=0.4, n_terms=25, seed=42):
    """w_k(x) via V_k(x) = Phi(mu_k + beta_k x): Gaussian-linked
    covariate-dependent sticks (sec. 14.9.3). Phi computed with
    erf. Keys: estimate."""
    rng = np.random.default_rng(seed)
    def Phi(v):
        return 0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))
    V = [Phi(float(rng.normal(0, 1))
             + float(rng.normal(0, 1)) * x)
         for _ in range(n_terms)]
    W = _bnp.stick_breaking(V)
    res = RichResult(payload={"estimate": W[0],
                              "total_mass": sum(W),
                              "method": "probit stick-breaking (GvdV 2017 sec. 14.9.3)"})
    return with_describe_pointer(res, "gh_c14_20")


def cheatsheet():
    return "gh_c14_20: Probit stick-breaking process"
