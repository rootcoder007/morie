# morie.fn -- function file (rootcoder007/morie)
"""Expectation propagation for GPs.

Implements sec. 11.7.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ep_gp"]


def _chol_solve(K, y):
    """Solve K x = y via the native linalg solver."""
    x = np.linalg.solve(np.marr(K), np.marr(y))
    return [float(v) for v in x._flat()]


def ghosal_ep_gp(x=None, y=None, length=0.5):
    """q(f) = N(mu, Sigma), Sigma = (K^{-1} + sum Lambda_i)^{-1}
    (sec. 11.7.4): Gaussian site approximations to the likelihood
    terms. Simplified EP: site precisions Lambda_i from the
    Laplace-style curvature at the current mean, iterated to a fixed
    point. Keys: estimate."""
    if x is None:
        x = [0.1, 0.3, 0.7, 0.9]
        y = [0.0, 0.0, 1.0, 1.0]
    xs = _bnp._flat(x)
    ys = _bnp._flat(y)
    n = len(xs)
    K = [[math.exp(-0.5 * ((xs[i] - xs[j]) / length) ** 2)
          + (1e-8 if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    mu = [0.0] * n
    Lam = [0.25] * n
    for _ in range(60):
        p = [1.0 / (1.0 + math.exp(-v)) for v in mu]
        Lam = [max(pi * (1.0 - pi), 1e-4) for pi in p]
        # mu solves (K^{-1} + Lam) mu = grad-like target: use
        # mu = K (y - p) damped toward the data pull
        pull = [ys[i] - p[i] for i in range(n)]
        Kpull = [sum(K[i][j] * pull[j] for j in range(n))
                 for i in range(n)]
        mu = [0.7 * m + 0.3 * (m + kp)
              for m, kp in zip(mu, Kpull)]
    # Sigma_00 = ((K^{-1} + diag(Lam))^{-1})_{00}
    Kinv_cols = [_chol_solve(K, [1.0 if r == c else 0.0
                                 for r in range(n)])
                 for c in range(n)]
    B = [[Kinv_cols[c][r] + (Lam[r] if r == c else 0.0)
          for c in range(n)] for r in range(n)]
    var0 = _chol_solve(B, [1.0] + [0.0] * (n - 1))[0]
    p = [1.0 / (1.0 + math.exp(-v)) for v in mu]
    res = RichResult(payload={"estimate": p[-1],
                              "site_precisions": Lam,
                              "ep_var_site0": var0,
                              "separates": p[-1] > 0.5 > p[0],
                              "method": "EP for GP classification (GvdV 2017 sec. 11.7.4)"})
    return with_describe_pointer(res, "gh_c11_15")


def cheatsheet():
    return "gh_c11_15: Expectation propagation for GPs"


# compact alias per ledger/NAMING.md
ghosalepgp = ghosal_ep_gp
