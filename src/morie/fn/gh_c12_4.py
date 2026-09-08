# morie.fn -- function file (rootcoder007/morie)
"""Semiparametric BvM for functionals.

Implements sec. 12.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_semipara_bvm"]


def ghosal_semipara_bvm(n=2000, alpha=2.0, n_sim=400, seed=42):
    """sqrt(n)(psi(G_post) - psi(F0)) -> N(0, sigma_eff^2) for a
    smooth functional (sec. 12.3): psi = mean of a uniform truth;
    the DP posterior-mean functional attains the efficient variance
    var(X) = 1/12. Keys: estimate."""
    rng = np.random.default_rng(seed)
    devs = []
    for _ in range(n_sim):
        s = 0.0
        for _ in range(n):
            s += float(rng.uniform(0, 1))
        post_mean = (alpha * 0.5 + s) / (alpha + n)
        devs.append(math.sqrt(n) * (post_mean - 0.5))
    m = sum(devs) / n_sim
    v = sum((d - m) ** 2 for d in devs) / (n_sim - 1)
    res = RichResult(payload={"estimate": v,
                              "efficient_variance": 1.0 / 12.0,
                              "gap": abs(v - 1.0 / 12.0),
                              "method": "semiparametric BvM (GvdV 2017 sec. 12.3)"})
    return with_describe_pointer(res, "gh_c12_4")


def cheatsheet():
    return "gh_c12_4: Semiparametric BvM for functionals"
