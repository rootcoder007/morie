# morie.fn -- function file (rootcoder007/morie)
"""NTR functional BvM.

Implements sec. 13.4.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ntr_bvm"]


def ghosal_ntr_bvm(n=1500, n_sim=300, seed=42):
    """sqrt(n)(psi(F_post) - psi(F0)) -> N(0, sigma^2) for smooth
    functionals of NTR posteriors (sec. 13.4.2). psi = F(1) under
    uncensored exponential truth: deviations scale at sqrt(n) with
    variance F0(1)(1 - F0(1)). Keys: estimate."""
    rng = np.random.default_rng(seed)
    F0_1 = 1.0 - math.exp(-1.0)
    devs = []
    for _ in range(n_sim):
        cnt = sum(1 for _ in range(n)
                  if -math.log(max(float(rng.uniform(0, 1)),
                                   1e-12)) <= 1.0)
        post = (2.0 * F0_1 + cnt) / (2.0 + n)
        devs.append(math.sqrt(n) * (post - F0_1))
    m = sum(devs) / n_sim
    v = sum((d - m) ** 2 for d in devs) / (n_sim - 1)
    target = F0_1 * (1.0 - F0_1)
    res = RichResult(payload={"estimate": v,
                              "efficient_variance": target,
                              "gap": abs(v - target),
                              "method": "NTR functional BvM (GvdV 2017 sec. 13.4.2)"})
    return with_describe_pointer(res, "gh_c13_11")


def cheatsheet():
    return "gh_c13_11: NTR functional BvM"


# compact alias per ledger/NAMING.md
ghosalntrbvm = ghosal_ntr_bvm
