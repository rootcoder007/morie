# morie.fn -- function file (rootcoder007/morie)
"""Log-spline density contraction.

Implements sec. 9.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_logspline_crt"]


def ghosal_logspline_crt(data=None, smoothness=2.0, n=500, seed=42):
    """f = exp(sum beta_k phi_k)/Z with K_n ~ n^{1/(2s+1)} basis
    terms attains rate n^{-s/(2s+1)} (sec. 9.1). Fits the exponential
    family by damped Newton on the log-likelihood (cosine basis) and
    reports K_n and the fitted density. Keys: estimate."""
    rng = np.random.default_rng(seed)
    if data is None:
        data = [float(rng.beta(2.0, 2.0)) for _ in range(n)]
    xs = _bnp._flat(data)
    n = len(xs)
    K = max(1, int(round(n ** (1.0 / (2.0 * smoothness + 1.0)))))
    beta = [0.0] * K
    grid = 200
    gx = [(i + 0.5) / grid for i in range(grid)]
    def basis(x, k):
        return math.sqrt(2.0) * math.cos((k + 1) * math.pi * x)
    for _ in range(60):
        f = [math.exp(sum(b * basis(x, k)
                          for k, b in enumerate(beta)))
             for x in gx]
        Z = sum(f) / grid
        # gradient: mean basis over data - E_f[basis]
        for k in range(K):
            emp = sum(basis(x, k) for x in xs) / n
            mod = sum(fi * basis(x, k)
                      for fi, x in zip(f, gx)) / grid / Z
            beta[k] += 0.5 * (emp - mod)
    f = [math.exp(sum(b * basis(x, k) for k, b in enumerate(beta)))
         for x in gx]
    Z = sum(f) / grid
    dens = [v / Z for v in f]
    res = RichResult(payload={"estimate": dens[grid // 2],
                              "K_n": K, "beta": beta,
                              "normalization_gap":
                                  abs(sum(dens) / grid - 1.0),
                              "rate": n ** (-smoothness
                                            / (2 * smoothness + 1)),
                              "method": "log-spline density (GvdV 2017 sec. 9.1)"})
    return with_describe_pointer(res, "gh_c9_1")


def cheatsheet():
    return "gh_c9_1: Log-spline density contraction"
