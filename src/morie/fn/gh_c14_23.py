# morie.fn -- function file (rootcoder007/morie)
"""Indian buffet process.

Implements sec. 14.10 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ibp_def"]


def ghosal_ibp_def(n_customers=30, alpha=3.0, seed=42):
    """IBP: customer i samples each existing dish k with probability
    m_k / i and Poisson(alpha / i) new dishes (sec. 14.10): the
    expected total number of dishes is alpha H_n. Keys: estimate."""
    rng = np.random.default_rng(seed)
    dish_counts = []
    def rpois(lam):
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            p *= float(rng.uniform(0, 1))
            if p <= L:
                return k
            k += 1
    for i in range(1, int(n_customers) + 1):
        for k in range(len(dish_counts)):
            if float(rng.uniform(0, 1)) < dish_counts[k] / i:
                dish_counts[k] += 1
        for _ in range(rpois(alpha / i)):
            dish_counts.append(1)
    Hn = sum(1.0 / i for i in range(1, int(n_customers) + 1))
    res = RichResult(payload={"estimate": float(len(dish_counts)),
                              "expected_dishes": alpha * Hn,
                              "method": "Indian buffet process (GvdV 2017 sec. 14.10)"})
    return with_describe_pointer(res, "gh_c14_23")


def cheatsheet():
    return "gh_c14_23: Indian buffet process"
