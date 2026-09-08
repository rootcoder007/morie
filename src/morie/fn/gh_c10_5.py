# morie.fn -- function file (rootcoder007/morie)
"""Spike-and-slab white-noise adaptation.

Implements sec. 10.3.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_wn_adapt"]


def ghosal_wn_adapt(y=None, n=400, pi_incl=0.2, tau2=1.0, seed=42):
    """theta_jk ~ pi N(0, tau_j^2) + (1 - pi) delta_0: exact posterior
    inclusion probabilities recover a sparse truth (sec. 10.3.2).
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    if y is None:
        truth = [1.2, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        y = [t + float(rng.normal(0, 1)) / math.sqrt(n)
             for t in truth]
    v = 1.0 / n
    incl = []
    for yk in y:
        l1 = -0.5 * math.log(2 * math.pi * (v + tau2)) \
            - 0.5 * yk * yk / (v + tau2) + math.log(pi_incl)
        l0 = -0.5 * math.log(2 * math.pi * v) \
            - 0.5 * yk * yk / v + math.log(1.0 - pi_incl)
        incl.append(1.0 / (1.0 + math.exp(l0 - l1)))
    res = RichResult(payload={"estimate": incl[0],
                              "inclusion_probs": incl,
                              "method": "spike-slab adaptation (GvdV 2017 sec. 10.3.2)"})
    return with_describe_pointer(res, "gh_c10_5")


def cheatsheet():
    return "gh_c10_5: Spike-and-slab white-noise adaptation"


# compact alias per ledger/NAMING.md
ghosalwnadapt = ghosal_wn_adapt
