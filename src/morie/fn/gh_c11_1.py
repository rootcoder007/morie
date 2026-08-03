# morie.fn -- function file (rootcoder007/morie)
"""Gaussian process RKHS (Euclidean case).

Implements Example 11.15 + Definition 11.12, eq. (11.8) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gp_def_rkhs"]


def _chol_solve(K, y):
    """Solve K x = y via the native linalg solver."""
    x = np.linalg.solve(np.marr(K), np.marr(y))
    return [float(v) for v in x._flat()]


def ghosal_gp_def_rkhs(Sigma, a, b):
    """For W ~ N(0, Sigma) the RKHS is the range of Sigma with inner
    product <Sigma a, Sigma b>_H = a' Sigma b (Ex 11.15); the
    reproducing formula h(t) = <h, K(t, .)>_H (eq. 11.8) is verified
    for h = Sigma a. Keys: estimate."""
    S = [[float(v) for v in row] for row in Sigma]
    a = _bnp._flat(a)
    b = _bnp._flat(b)
    k = len(a)
    ip = sum(a[i] * S[i][j] * b[j] for i in range(k)
             for j in range(k))
    h = [sum(S[i][j] * a[j] for j in range(k)) for i in range(k)]
    # reproducing check at coordinate t: <h, K(t,.)>_H with
    # K(t,.) = Sigma e_t, coefficients e_t: = a' Sigma e_t = h(t)
    gaps = [abs(sum(a[i] * S[i][t] for i in range(k)) - h[t])
            for t in range(k)]
    res = RichResult(payload={"estimate": ip,
                              "h": h,
                              "reproducing_gap": max(gaps),
                              "method": "Euclidean RKHS (GvdV 2017 Ex 11.15, eq. 11.8)"})
    return with_describe_pointer(res, "gh_c11_1")


def cheatsheet():
    return "gh_c11_1: Gaussian process RKHS (Euclidean case)"
