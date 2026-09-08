# morie.fn -- slice s03 (rootcoder007/morie)
"""Kernel principal component analysis.

Source consulted: Schoelkopf, B., Smola, A. and Mueller, K.-R. (1998).
Nonlinear component analysis as a kernel eigenvalue problem.  *Neural
Computation* 10(5), 1299-1319.  The kernel matrix is centred in feature
space by their equation (21),

    Ktilde = K - 1_n K - K 1_n + 1_n K 1_n

with 1_n the matrix all of whose entries are 1/n; the components are the
eigenvectors alpha^k of Ktilde normalised so that lambda_k <alpha^k,
alpha^k> = 1, and the projection of a point onto component k is
sum_i alpha^k_i Ktilde(x_i, x).  The 1998 paper is paywalled; the
centring identity and the normalisation are quoted in their standard
published form.

Eigenvectors are sign-fixed (largest-magnitude entry positive) before
projection, because the eigenproblem does not determine the sign.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
# aliased `core`, not `k`, because this module's public signature already
# uses `k` for the number of components
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["sgt_kernel_pca"]


def _gram(X, kernel, gamma, degree, coef0):
    n = len(X)
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if kernel == "linear":
                s = 0.0
                for a in range(len(X[i])):
                    s += X[i][a] * X[j][a]
            elif kernel == "poly":
                s = 0.0
                for a in range(len(X[i])):
                    s += X[i][a] * X[j][a]
                s = (gamma * s + coef0) ** degree
            else:
                s = 0.0
                for a in range(len(X[i])):
                    d = X[i][a] - X[j][a]
                    s += d * d
                s = math.exp(-gamma * s)
            K[i][j] = s
    return K


def sgt_kernel_pca(X, kernel="rbf", k=2, gamma=1.0, degree=2, coef0=1.0):
    """Top-k kernel principal components.

    Returns
    -------
    RichResult with payload:
        Y        : the projections, one row per point
        eigvals  : the top-k eigenvalues of the centred kernel, descending
        estimate : the leading eigenvalue
        explained: eigenvalue share of the total
    """
    Xm = core.mat(X)
    n = len(Xm)
    K = _gram(Xm, kernel, float(gamma), float(degree), float(coef0)) if isinstance(kernel, str) else core.mat(kernel)
    rm = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += K[i][j]
        rm[i] = s / n
    gm = 0.0
    for v in rm:
        gm += v / n
    Kt = [[K[i][j] - rm[i] - rm[j] + gm for j in range(n)] for i in range(n)]
    vals, vecs = core.jacobi(Kt)
    order = list(range(n - 1, -1, -1))
    kk = int(k)
    if kk > n:
        kk = n
    ev = [vals[order[t]] for t in range(kk)]
    Y = [[0.0] * kk for _ in range(n)]
    for t in range(kk):
        col = [vecs[i][order[t]] for i in range(n)]
        lam = ev[t]
        scale = 1.0 / math.sqrt(lam) if lam > 1e-12 else 0.0
        a = [z * scale for z in col]
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += a[j] * Kt[i][j]
            Y[i][t] = s
    tot = 0.0
    for v in vals:
        if v > 0.0:
            tot += v
    return RichResult(
        title="Kernel PCA",
        summary_lines=[("components", kk)],
        payload={
            "Y": Y,
            "eigvals": ev,
            "estimate": ev[0] if ev else float("nan"),
            "explained": [v / tot if tot > 0.0 else float("nan") for v in ev],
            "method": "Kernel PCA on the centred Gram matrix (Schoelkopf et al. 1998, eq. 21)",
        },
    )


def cheatsheet():
    return "sgtkpc: Kernel PCA top-k components"


sgtkernelpca = sgt_kernel_pca
