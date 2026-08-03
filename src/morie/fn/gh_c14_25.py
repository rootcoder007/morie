# morie.fn -- function file (rootcoder007/morie)
"""IBP as a Poisson feature process.

Implements sec. 14.10 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ibp_poisson"]


def ghosal_ibp_poisson(n_customers=25, alpha=3.0, n_sim=300,
                       seed=42):
    """New-dish counts are independent Poisson(alpha / i): the total
    K_n ~ Poisson(alpha H_n) exactly (sec. 14.10). Monte Carlo mean
    against alpha H_n. Keys: estimate."""
    rng = np.random.default_rng(seed)
    def rpois(lam):
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            p *= float(rng.uniform(0, 1))
            if p <= L:
                return k
            k += 1
    Hn = sum(1.0 / i for i in range(1, int(n_customers) + 1))
    tot = 0.0
    for _ in range(n_sim):
        K = sum(rpois(alpha / i)
                for i in range(1, int(n_customers) + 1))
        tot += K / n_sim
    res = RichResult(payload={"estimate": tot,
                              "theory": alpha * Hn,
                              "gap": abs(tot - alpha * Hn),
                              "method": "IBP Poisson representation (GvdV 2017 sec. 14.10)"})
    return with_describe_pointer(res, "gh_c14_25")


def cheatsheet():
    return "gh_c14_25: IBP as a Poisson feature process"
