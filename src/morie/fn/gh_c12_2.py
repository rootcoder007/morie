# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet-process BvM (Brownian bridge).

Implements sec. 12.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_bvm"]


def ghosal_dp_bvm(n=2000, alpha=2.0, n_sim=400, seed=42):
    """sqrt(n)(F_post - F0) converges to B(F0) for B a Brownian
    bridge (sec. 12.2): at a fixed t the limit is N(0, F0(t)(1 -
    F0(t))). Simulated posterior-mean deviations match the bridge
    variance. Keys: estimate."""
    rng = np.random.default_rng(seed)
    t = 0.3
    devs = []
    for _ in range(n_sim):
        cnt = sum(1 for _ in range(n)
                  if float(rng.uniform(0, 1)) <= t)
        post = (alpha * t + cnt) / (alpha + n)
        devs.append(math.sqrt(n) * (post - t))
    m = sum(devs) / n_sim
    v = sum((d - m) ** 2 for d in devs) / (n_sim - 1)
    v_bridge = t * (1.0 - t)
    res = RichResult(payload={"estimate": v,
                              "bridge_variance": v_bridge,
                              "gap": abs(v - v_bridge),
                              "method": "DP BvM / Brownian bridge (GvdV 2017 sec. 12.2)"})
    return with_describe_pointer(res, "gh_c12_2")


def cheatsheet():
    return "gh_c12_2: Dirichlet-process BvM (Brownian bridge)"


# compact alias per ledger/NAMING.md
ghosaldpbvm = ghosal_dp_bvm
