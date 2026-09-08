# morie.fn -- function file (rootcoder007/morie)
"""Parametric Bernstein-von Mises.

Implements sec. 12.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_infdim_bvm"]


def ghosal_infdim_bvm(theta0=0.4, n=2000, seed=42):
    """sqrt(n)-rescaled posterior approaches N(theta-hat, I^{-1}/n)
    in total variation (sec. 12.1). Beta-Bernoulli: exact posterior
    vs the BvM normal -- total variation on a grid shrinks.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    S = sum(1 for _ in range(n)
            if float(rng.uniform(0, 1)) < theta0)
    a, b = 1.0 + S, 1.0 + n - S
    mle = S / n
    I_inv = mle * (1.0 - mle)
    sd = math.sqrt(I_inv / n)
    grid = 2000
    tv = 0.0
    lo, hi = max(mle - 6 * sd, 1e-9), min(mle + 6 * sd, 1 - 1e-9)
    for i in range(grid):
        t = lo + (hi - lo) * (i + 0.5) / grid
        bpdf = math.exp(math.lgamma(a + b) - math.lgamma(a)
                        - math.lgamma(b) + (a - 1) * math.log(t)
                        + (b - 1) * math.log(1 - t))
        npdf = math.exp(-0.5 * ((t - mle) / sd) ** 2) \
            / (sd * math.sqrt(2 * math.pi))
        tv += 0.5 * abs(bpdf - npdf) * (hi - lo) / grid
    res = RichResult(payload={"estimate": tv,
                              "bvm_holds": tv < 0.05,
                              "method": "parametric BvM (GvdV 2017 sec. 12.1)"})
    return with_describe_pointer(res, "gh_c12_1")


def cheatsheet():
    return "gh_c12_1: Parametric Bernstein-von Mises"
