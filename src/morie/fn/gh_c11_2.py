# morie.fn -- function file (rootcoder007/morie)
"""Concentration-function terms.

Implements eq. (11.11) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_rkhs_norm"]


def ghosal_rkhs_norm(f0_coefs, lambdas, eps, n_sim=3000, seed=42):
    """phi_{w0}(eps) = inf{||h||_H^2/2: ||h - w0|| <= eps}
    - log P(||W|| < eps) (eq. 11.11) for a diagonal series GP with
    coordinate variances lambda_i: the infimum is attained by
    truncating/shrinking coordinates, the small-ball term by Monte
    Carlo on the l2 norm. Keys: estimate."""
    f0 = _bnp._flat(f0_coefs)
    lam = _bnp._flat(lambdas)
    # decentering: greedily match largest coordinates until within eps
    order = sorted(range(len(f0)), key=lambda i: -abs(f0[i]))
    h = [0.0] * len(f0)
    def resid():
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(f0, h)))
    hn2 = 0.0
    for i in order:
        if resid() <= eps:
            break
        h[i] = f0[i]
        hn2 += f0[i] ** 2 / lam[i]
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        s = sum(lam[i] * float(rng.normal(0, 1)) ** 2
                for i in range(len(lam)))
        if math.sqrt(s) < eps:
            hits += 1
    small_ball = -math.log(max(hits, 1) / n_sim)
    phi = 0.5 * hn2 + small_ball
    res = RichResult(payload={"estimate": phi,
                              "decentering_norm2": hn2,
                              "small_ball_exponent": small_ball,
                              "method": "concentration function (GvdV 2017 eq. 11.11)"})
    return with_describe_pointer(res, "gh_c11_2")


def cheatsheet():
    return "gh_c11_2: Concentration-function terms"


# compact alias per ledger/NAMING.md
ghosalrkhsnorm = ghosal_rkhs_norm
