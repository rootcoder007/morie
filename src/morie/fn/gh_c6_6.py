# morie.fn -- function file (rootcoder007/morie)
"""Kullback-Leibler support.

Implements Definition 6.15 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_kl_support"]


def ghosal_kl_support(p0, alpha=None, eps=0.1, n_sim=3000, seed=42):
    """p0 in KL(Pi) iff Pi(p: K(p0; p) < eps) > 0 for every eps
    (Def 6.15): Monte Carlo estimate of the prior mass of the KL
    neighborhood under a Dirichlet prior on the k-cell simplex.
    Keys: estimate."""
    p0 = _bnp.normalize_weights(p0)
    k = len(p0)
    if alpha is None:
        alpha = [1.0] * k
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        g = [float(rng.gamma(a, 1.0)) for a in alpha]
        p = _bnp.normalize_weights(g)
        kl = sum(q * math.log(q / max(pi, 1e-300))
                 for q, pi in zip(p0, p) if q > 0)
        if kl < eps:
            hits += 1
    mass = hits / n_sim
    res = RichResult(payload={"estimate": mass,
                              "kl_property": mass > 0,
                              "method": "KL support mass (GvdV 2017 Def 6.15)"})
    return with_describe_pointer(res, "gh_c6_6")


def cheatsheet():
    return "gh_c6_6: Kullback-Leibler support"
