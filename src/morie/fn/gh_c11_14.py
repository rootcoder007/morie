# morie.fn -- function file (rootcoder007/morie)
"""Laplace approximation for GP posteriors.

Implements sec. 11.7.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gp_laplace"]


def _chol_solve(K, y):
    """Solve K x = y via the native linalg solver."""
    x = np.linalg.solve(np.marr(K), np.marr(y))
    return [float(v) for v in x._flat()]


def ghosal_gp_laplace(x=None, y=None, length=0.5, seed=42):
    """pi(f | data) ~ N(f-hat, (K^{-1} + W)^{-1}) with W =
    diag(-d^2 log p(y|f)) at the mode (sec. 11.7.5). GP logistic
    classification: damped Newton finds f-hat; the Laplace variance
    uses W = diag(p(1-p)). Keys: estimate."""
    if x is None:
        x = [0.1, 0.25, 0.4, 0.6, 0.75, 0.9]
        y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    xs = _bnp._flat(x)
    ys = _bnp._flat(y)
    n = len(xs)
    K = [[math.exp(-0.5 * ((xs[i] - xs[j]) / length) ** 2)
          + (1e-8 if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    f = [0.0] * n
    for _ in range(100):
        p = [1.0 / (1.0 + math.exp(-v)) for v in f]
        grad_data = [yi - pi for yi, pi in zip(ys, p)]
        Kinv_f = _chol_solve(K, f)
        step = [0.3 * (sum(K[i][j] * grad_data[j]
                           for j in range(n)) - f[i])
                for i in range(n)]
        f = [fi + si for fi, si in zip(f, step)]
    p = [1.0 / (1.0 + math.exp(-v)) for v in f]
    W = [pi * (1.0 - pi) for pi in p]
    # Laplace variance at site 0: ((K^{-1} + W)^{-1})_{00} via solve
    A = [[(1.0 if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    # A = K^{-1} + W: build via solving K columns
    Kinv_cols = [_chol_solve(K, [1.0 if r == c else 0.0
                                 for r in range(n)])
                 for c in range(n)]
    B = [[Kinv_cols[c][r] + (W[r] if r == c else 0.0)
          for c in range(n)] for r in range(n)]
    e0 = [1.0] + [0.0] * (n - 1)
    var0 = _chol_solve(B, e0)[0]
    res = RichResult(payload={"estimate": p[-1],
                              "mode_probs": p,
                              "laplace_var_site0": var0,
                              "separates": p[-1] > 0.5 > p[0],
                              "method": "GP Laplace approximation (GvdV 2017 sec. 11.7.5)"})
    return with_describe_pointer(res, "gh_c11_14")


def cheatsheet():
    return "gh_c11_14: Laplace approximation for GP posteriors"
