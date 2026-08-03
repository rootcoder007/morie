# morie.fn -- function file (rootcoder007/morie)
"""White-noise linear-functional BvM.

Implements sec. 12.4.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_wn_lin_bvm"]


def ghosal_wn_lin_bvm(L_coefs=(0.6, 0.8), n=500, prior_var=50.0,
                      n_sim=500, seed=42):
    """sqrt(n)(L(theta_post) - L(theta0)) -> N(0, ||L||^2) for a
    bounded linear functional (sec. 12.4.2). Conjugate simulation:
    the rescaled posterior-mean functional has variance ~ ||L||^2.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    L = _bnp._flat(L_coefs)
    L2 = sum(v * v for v in L)
    theta0 = [0.3, -0.4]
    devs = []
    shrink = prior_var / (prior_var + 1.0 / n)
    for _ in range(n_sim):
        y = [t + float(rng.normal(0, 1)) / math.sqrt(n)
             for t in theta0]
        post = [shrink * v for v in y]
        devs.append(math.sqrt(n) * sum(
            l * (p - t) for l, p, t in zip(L, post, theta0)))
    m = sum(devs) / n_sim
    v = sum((d - m) ** 2 for d in devs) / (n_sim - 1)
    res = RichResult(payload={"estimate": v,
                              "norm2_L": L2,
                              "gap": abs(v - L2),
                              "method": "linear-functional BvM (GvdV 2017 sec. 12.4.2)"})
    return with_describe_pointer(res, "gh_c12_10")


def cheatsheet():
    return "gh_c12_10: White-noise linear-functional BvM"
