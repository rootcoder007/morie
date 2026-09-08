# morie.fn -- function file (rootcoder007/morie)
"""Stationary GP via Bochner.

Implements Example 11.8, eq. (11.3)-(11.4) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_statgp_spec"]


def ghosal_statgp_spec(h=0.6, n_grid=4000, lam_max=30.0):
    """K(s - t) = int e^{-i <s-t, lambda>} dmu(lambda) (eq. 11.3);
    the square-exponential process has dmu = (2 sqrt(pi))^{-1}
    e^{-lambda^2/4} dlambda in d = 1 (eq. 11.4) and kernel
    K(h) = e^{-h^2}. Numeric inversion of the spectral integral.
    Keys: estimate."""
    tot = 0.0
    step = 2.0 * lam_max / n_grid
    for i in range(n_grid):
        lam = -lam_max + (i + 0.5) * step
        tot += math.cos(h * lam) * math.exp(-lam * lam / 4.0) * step
    K_num = tot / (2.0 * math.sqrt(math.pi))
    K_true = math.exp(-h * h)
    res = RichResult(payload={"estimate": K_num,
                              "kernel_exact": K_true,
                              "bochner_gap": abs(K_num - K_true),
                              "method": "Bochner spectral kernel (GvdV 2017 eq. 11.3-11.4)"})
    return with_describe_pointer(res, "gh_c11_9")


def cheatsheet():
    return "gh_c11_9: Stationary GP via Bochner"
