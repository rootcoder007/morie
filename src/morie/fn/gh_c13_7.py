# morie.fn -- function file (rootcoder007/morie)
"""Mixtures of beta processes.

Implements sec. 13.3.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_mix_bp"]


def ghosal_mix_bp(lambdas=(0.5, 1.0, 2.0), weights=None, c=3.0,
                  t=1.0):
    """H ~ int BP(c, H0_lambda) dPi(lambda): the prior mean hazard
    mixes the component means, E H(t) = sum w_j H0_{lambda_j}(t)
    (sec. 13.3.4). Keys: estimate."""
    ls = _bnp._flat(lambdas)
    if weights is None:
        weights = [1.0 / len(ls)] * len(ls)
    w = _bnp.normalize_weights(weights)
    mean_H = sum(wi * li * t for wi, li in zip(w, ls))
    res = RichResult(payload={"estimate": mean_H,
                              "component_means": [li * t
                                                  for li in ls],
                              "method": "mixture of beta processes (GvdV 2017 sec. 13.3.4)"})
    return with_describe_pointer(res, "gh_c13_7")


def cheatsheet():
    return "gh_c13_7: Mixtures of beta processes"


# compact alias per ledger/NAMING.md
ghosalmixbp = ghosal_mix_bp
