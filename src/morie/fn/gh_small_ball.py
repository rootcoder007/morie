# morie.fn -- function file (rootcoder007/morie)
"""Small-ball exponent.

Implements eq. (11.10) + Lemma 11.27 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_small_ball_prob"]


def ghosal_small_ball_prob(eps_list=(1.0, 0.5, 0.25), n_grid=64,
                           n_sim=1500, seed=42):
    """P(||W||_infty < eps) = e^{-phi_0(eps)} (eq. 11.10); for
    Brownian motion phi_0(eps) ~ eps^{-2} (Lemma 11.27). Monte Carlo
    on discretized BM paths; the exponent ratio between eps and
    eps/2 should approach 4. Keys: estimate."""
    rng = np.random.default_rng(seed)
    phis = []
    for eps in eps_list:
        hits = 0
        for _ in range(n_sim):
            w = 0.0
            ok = True
            for _ in range(n_grid):
                w += float(rng.normal(0, 1)) / math.sqrt(n_grid)
                if abs(w) >= eps:
                    ok = False
                    break
            hits += 1 if ok else 0
        phis.append(-math.log(max(hits, 1) / n_sim))
    res = RichResult(payload={"estimate": phis[-1],
                              "phi_by_eps": phis,
                              "increasing": all(
                                  phis[i + 1] >= phis[i] - 1e-9
                                  for i in range(len(phis) - 1)),
                              "method": "BM small-ball exponent (GvdV 2017 eq. 11.10, Lemma 11.27)"})
    return with_describe_pointer(res, "gh_small_ball")


def cheatsheet():
    return "gh_small_ball: Small-ball exponent"
