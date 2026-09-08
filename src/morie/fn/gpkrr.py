# morie.fn -- slice s03 (rootcoder007/morie)
"""Kernel ridge regression, the dual of a Gaussian-process mean.

Source consulted: Saunders, C., Gammerman, A. and Vovk, V. (1998).
Ridge regression learning algorithm in dual variables.  *ICML* 15,
515-521.  Their dual solution is

    alpha = (K + lambda I)^(-1) y,   fhat(x) = sum_i alpha_i k(x_i, x)

The 1998 proceedings were not retrievable here; the two expressions are
quoted in their standard published form.  The identity worth stating is
that this is *exactly* the posterior mean of a Gaussian process with
covariance k and noise variance lambda (Rasmussen and Williams 2006,
eq. 2.23), which is why the module is named for the GP: the two
procedures differ only in that the GP also returns a variance, and that
variance is returned here as well.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["gp_kernel_ridge_reg"]


def _rbf(x, y, gamma):
    s = 0.0
    for a in range(len(x)):
        d = x[a] - y[a]
        s += d * d
    return math.exp(-gamma * s)


def gp_kernel_ridge_reg(X, y, X_test=None, lam=1e-2, gamma=1.0):
    """Dual ridge / GP posterior mean and variance.

    Returns
    -------
    RichResult with payload:
        estimate : prediction at the first test point
        pred     : predictions at every test point
        var      : GP posterior variance at every test point
        alpha    : the dual coefficients
    """
    Xm = k.mat(X)
    yv = k.vec(y)
    n = len(Xm)
    K = [[_rbf(Xm[i], Xm[j], float(gamma)) for j in range(n)] for i in range(n)]
    A = [[K[i][j] + (float(lam) if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    alpha = k.cholsolve(A, yv)
    Xt = k.mat(X_test) if X_test is not None else Xm
    pred = []
    var = []
    for t in range(len(Xt)):
        ks = [_rbf(Xm[i], Xt[t], float(gamma)) for i in range(n)]
        s = 0.0
        for i in range(n):
            s += alpha[i] * ks[i]
        pred.append(s)
        w = k.cholsolve(A, ks)
        q = 0.0
        for i in range(n):
            q += ks[i] * w[i]
        var.append(_rbf(Xt[t], Xt[t], float(gamma)) - q)
    return RichResult(
        title="Kernel ridge regression",
        summary_lines=[("test points", len(Xt))],
        payload={
            "estimate": pred[0] if pred else float("nan"),
            "pred": pred,
            "var": var,
            "alpha": alpha,
            "lam": float(lam),
            "method": "Dual ridge alpha = (K + lambda I)^-1 y (Saunders et al. 1998); equals the GP posterior mean",
        },
    )


def cheatsheet():
    return "gpkrr: GP-equivalent kernel ridge regression"
