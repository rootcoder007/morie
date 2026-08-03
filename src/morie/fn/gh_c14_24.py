# morie.fn -- function file (rootcoder007/morie)
"""IBP stick-breaking representation.

Implements sec. 14.10 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ibp_stickbr"]


def ghosal_ibp_stickbr(n_feats=40, alpha=2.0, seed=42):
    """pi_k = prod_{j<=k} V_j with V_j iid Beta(alpha, 1):
    feature probabilities DECAY multiplicatively; P(Z_{ik}=1) = pi_k
    (sec. 14.10). Keys: estimate."""
    rng = np.random.default_rng(seed)
    pi = []
    cur = 1.0
    for _ in range(int(n_feats)):
        cur *= float(rng.beta(alpha, 1.0))
        pi.append(cur)
    res = RichResult(payload={"estimate": pi[0],
                              "pi_head": pi[:8],
                              "decreasing": all(
                                  pi[i + 1] <= pi[i] + 1e-15
                                  for i in range(len(pi) - 1)),
                              "expected_sum": alpha,
                              "method": "IBP stick-breaking (GvdV 2017 sec. 14.10)"})
    return with_describe_pointer(res, "gh_c14_24")


def cheatsheet():
    return "gh_c14_24: IBP stick-breaking representation"
