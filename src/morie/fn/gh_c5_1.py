# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet process mixture model.

Implements eq. (5.1)-(5.2) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dpm_model"]


def _norm_pdf(x, mu, sd):
    z = (x - mu) / sd
    return math.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def ghosal_dpm_model(x, alpha=1.0, n_terms=200, kernel_sd=0.25,
                     seed=42):
    """p_F(x) = int psi(x; theta) dF(theta), F ~ DP(alpha) (eq. 5.1):
    realized by the stick-breaking series sum W_j psi(x; theta_j)
    with a normal kernel. Keys: estimate."""
    rng = np.random.default_rng(seed)
    M = float(alpha)
    V = [float(rng.beta(1.0, M)) for _ in range(n_terms)]
    W = _bnp.stick_breaking(V)
    th = [float(v) for v in rng.uniform(0, 1, n_terms)._flat()]
    xs = _bnp._flat(x)
    dens = [sum(w * _norm_pdf(xi, t, kernel_sd)
                for w, t in zip(W, th)) for xi in xs]
    res = RichResult(payload={"estimate": dens[0], "density": dens,
                              "mixing_mass": sum(W),
                              "method": "DP mixture density (GvdV 2017 eq. 5.1)"})
    return with_describe_pointer(res, "gh_c5_1")


def cheatsheet():
    return "gh_c5_1: Dirichlet process mixture model"


# compact alias per ledger/NAMING.md
ghosaldpmmodel = ghosal_dpm_model
