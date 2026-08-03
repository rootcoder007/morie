# morie.fn -- function file (rootcoder007/morie)
"""Characterization of the DP by neutrality.

Implements Theorem 4.28 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_charact"]


def ghosal_dp_charact(dir_params, n_sim=4000, seed=42):
    """P is a DP iff it is neutral: P(A_1) is independent of the
    normalized rest (P(A_2)/(1-P(A_1)), ...) (Theorem 4.28). Checked
    by simulation: the sample correlation between P(A_1) and the
    renormalized second cell vanishes for a Dirichlet vector.
    Keys: estimate."""
    a = _bnp._flat(dir_params)
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    for _ in range(n_sim):
        g = [float(rng.gamma(ai, 1.0)) for ai in a]
        p = _bnp.normalize_weights(g)
        xs.append(p[0])
        ys.append(p[1] / (1.0 - p[0]))
    mx = sum(xs) / n_sim
    my = sum(ys) / n_sim
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    corr = sxy / (sx * sy)
    res = RichResult(payload={"estimate": corr,
                              "neutral": abs(corr) < 0.05,
                              "method": "DP neutrality check (GvdV 2017 Thm 4.28)"})
    return with_describe_pointer(res, "gh_c4_19")


def cheatsheet():
    return "gh_c4_19: Characterization of the DP by neutrality"
