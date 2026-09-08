# morie.fn -- function file (rootcoder007/morie)
"""Prior-mass condition neighborhoods.

Implements eq. (8.3)-(8.4) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_prior_mass_cnd"]


def ghosal_prior_mass_cnd(p0, eps=0.3, alpha=None, n_sim=4000,
                          seed=42):
    """B_2(p0, eps) = {p: K(p0; p) < eps^2, V_{2,0}(p0; p) < eps^2}
    (eq. 8.3): Monte Carlo prior mass under a Dirichlet prior --
    condition (8.4) needs it >= e^{-C n eps_n^2}. Keys: estimate."""
    p0 = _bnp.normalize_weights(p0)
    k = len(p0)
    if alpha is None:
        alpha = [1.0] * k
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        g = [float(rng.gamma(a, 1.0)) for a in alpha]
        p = _bnp.normalize_weights(g)
        lr = [math.log(q / max(pi, 1e-300))
              for q, pi in zip(p0, p)]
        K = sum(q * l for q, l in zip(p0, lr))
        V = sum(q * max(l - K, 0.0) ** 2 for q, l in zip(p0, lr))
        if K < eps ** 2 and V < eps ** 2:
            hits += 1
    mass = max(hits, 0) / n_sim
    res = RichResult(payload={"estimate": mass,
                              "log_mass": math.log(max(mass, 1e-12)),
                              "positive": mass > 0,
                              "method": "B_2 prior mass (GvdV 2017 eq. 8.3-8.4)"})
    return with_describe_pointer(res, "gh_c8_4")


def cheatsheet():
    return "gh_c8_4: Prior-mass condition neighborhoods"
