# morie.fn -- function file (rootcoder007/morie)
"""DP by gamma-process normalization.

Implements sec. 4.2.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_gamma"]


def ghosal_dp_gamma(base_masses, seed=42):
    """U(A_i) ~ Ga(alpha(A_i), 1) independently, P(A) = U(A)/U(X)
    (sec. 4.2.3): the completely-random-measure construction of the
    DP. Keys: estimate."""
    a = _bnp._flat(base_masses)
    rng = np.random.default_rng(seed)
    U = [float(rng.gamma(max(ai, 1e-12), 1.0)) for ai in a]
    tot = sum(U)
    p = [u / tot for u in U]
    res = RichResult(payload={"estimate": p[0], "P": p,
                              "gamma_total": tot,
                              "method": "gamma-process construction (GvdV 2017 sec. 4.2.3)"})
    return with_describe_pointer(res, "gh_c4_9")


def cheatsheet():
    return "gh_c4_9: DP by gamma-process normalization"


# compact alias per ledger/NAMING.md
ghosaldpgamma = ghosal_dp_gamma
