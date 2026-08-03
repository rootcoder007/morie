# morie.fn -- function file (rootcoder007/morie)
"""Local Dirichlet process.

Implements sec. 14.9.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_local_dp"]


def ghosal_local_dp(x=0.3, bandwidth=0.25, n_atoms=200, alpha=2.0,
                    seed=42):
    """G(x, .) = sum_k w_k(x) delta_{theta_k}: only atoms whose
    locations fall within the window of x get stick weight
    (sec. 14.9.2). Keys: estimate."""
    rng = np.random.default_rng(seed)
    locs = [float(rng.uniform(0, 1)) for _ in range(n_atoms)]
    Vs = [float(rng.beta(1.0, alpha)) for _ in range(n_atoms)]
    active = [i for i in range(n_atoms)
              if abs(locs[i] - x) <= bandwidth]
    W = _bnp.stick_breaking([Vs[i] for i in active])
    res = RichResult(payload={"estimate": sum(W),
                              "n_active": len(active),
                              "local": len(active) < n_atoms,
                              "method": "local DP (GvdV 2017 sec. 14.9.2)"})
    return with_describe_pointer(res, "gh_c14_19")


def cheatsheet():
    return "gh_c14_19: Local Dirichlet process"
