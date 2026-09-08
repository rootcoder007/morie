# morie.fn -- function file (rootcoder007/morie)
"""Credible-set frequentist coverage.

Implements sec. 12.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_cred_set_cov"]


def ghosal_cred_set_cov(theta0=0.5, n=400, level=0.9, n_sim=400,
                        seed=42):
    """When BvM holds, (1-alpha)-credible sets have frequentist
    coverage -> 1-alpha (sec. 12.5). Beta-Bernoulli central credible
    intervals via normal approximation to the Beta; empirical
    coverage near the level. Keys: estimate."""
    rng = np.random.default_rng(seed)
    z = 1.6448536269514722 if abs(level - 0.9) < 1e-9 else 1.96
    hits = 0
    for _ in range(n_sim):
        S = sum(1 for _ in range(n)
                if float(rng.uniform(0, 1)) < theta0)
        a, b = 1.0 + S, 1.0 + n - S
        m = a / (a + b)
        sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        if m - z * sd <= theta0 <= m + z * sd:
            hits += 1
    cov = hits / n_sim
    res = RichResult(payload={"estimate": cov,
                              "nominal": level,
                              "gap": abs(cov - level),
                              "method": "credible-set coverage (GvdV 2017 sec. 12.5)"})
    return with_describe_pointer(res, "gh_c12_11")


def cheatsheet():
    return "gh_c12_11: Credible-set frequentist coverage"
