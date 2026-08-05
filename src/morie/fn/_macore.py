# morie.fn -- function file (rootcoder007/morie)
"""Shared kernels for the meta-analysis modules.

Weighted least squares with its covariance, and the contrast design
matrix of a treatment network.  Both are needed by more than one module
and neither is worth writing twice.
"""

from . import _s03core as core

__all__ = ["wls", "net_design"]


def wls(X, y, w):
    """Weighted least squares; returns ``(beta, cov)``.

    ``cov`` is ``(X' W X)^{-1}``, the model-based covariance -- correct
    when the weights really are inverse variances, which is the whole
    premise of inverse-variance meta-analysis.
    """
    n = len(X)
    p = len(X[0])
    A = [[sum(w[i] * X[i][r] * X[i][s] for i in range(n)) for s in range(p)]
         for r in range(p)]
    b = [sum(w[i] * X[i][r] * y[i] for i in range(n)) for r in range(p)]
    beta = core.ridgesolve(A, b, 1e-12)
    cols = []
    for j in range(p):
        e = [1.0 if r == j else 0.0 for r in range(p)]
        cols.append(core.ridgesolve(A, e, 1e-12))
    cov = [[cols[j][r] for j in range(p)] for r in range(p)]
    return beta, cov, A


def net_design(design):
    """Contrast design matrix of a treatment network.

    Treatments are sorted and the smallest is the reference, whose effect
    is fixed at zero; a study comparing ``t2`` against ``t1`` contributes
    ``+1`` in the column of ``t2`` and ``-1`` in the column of ``t1``.
    Returns ``(X, treatments, T)``.
    """
    D = core.mat(design)
    if len(D[0]) != 2:
        raise ValueError("design must have two columns: baseline, comparator")
    treats = sorted({int(D[i][j]) for i in range(len(D)) for j in range(2)})
    pos = {t: j for j, t in enumerate(treats)}
    T = len(treats)
    if T < 2:
        raise ValueError("a network needs at least two treatments")
    X = []
    for row in D:
        r = [0.0] * (T - 1)
        b = pos[int(row[0])]
        c = pos[int(row[1])]
        if b > 0:
            r[b - 1] -= 1.0
        if c > 0:
            r[c - 1] += 1.0
        X.append(r)
    return X, treats, T
