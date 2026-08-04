# morie.fn -- function file (rootcoder007/morie)
"""Functional linear regression.

Implements sec. 10.4.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_func_reg"]


def ghosal_func_reg(n=300, K=4, seed=42):
    """E(Y | X) = int X(t) beta(t) dt with beta under a series prior
    (sec. 10.4.5): with X expanded on the same basis this is ridge
    regression on the coefficient scores; recovers beta0.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    beta0 = [1.0, -0.5, 0.25, 0.0]
    scores = []
    ys = []
    for _ in range(n):
        s = [float(rng.normal(0, 1)) for _ in range(K)]
        scores.append(s)
        ys.append(sum(b * v for b, v in zip(beta0, s))
                  + 0.2 * float(rng.normal(0, 1)))
    # ridge normal equations, diagonal-dominant so Gauss-Seidel
    bhat = [0.0] * K
    for _ in range(300):
        for k in range(K):
            num = sum((y - sum(bhat[j] * s[j] for j in range(K)
                               if j != k)) * s[k]
                      for s, y in zip(scores, ys))
            den = sum(s[k] * s[k] for s in scores) + 1.0
            bhat[k] = num / den
    err = max(abs(a - b) for a, b in zip(bhat, beta0))
    res = RichResult(payload={"estimate": err, "beta_hat": bhat,
                              "method": "functional regression (GvdV 2017 sec. 10.4.5)"})
    return with_describe_pointer(res, "gh_c10_11")


def cheatsheet():
    return "gh_c10_11: Functional linear regression"


# compact alias per ledger/NAMING.md
ghosalfuncreg = ghosal_func_reg
