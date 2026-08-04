# morie.fn -- function file (rootcoder007/morie)
"""Functional clustering by k-means on B-spline coefficients."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['fdaclust', 'functional_clustering']


def fdaclust(Y, K=2, basis=None, iters=25):
    """Functional clustering by k-means on B-spline coefficients.

    Two curves are close when their smooth parts are close rather than when their noise lines up. Initialisation is deterministic (the K curves at evenly spaced ranks of the first coefficient) and the loop runs a fixed iteration count with no tolerance exit, which is what lets the R arm reproduce the same labels.


    Formula: c_i = argmin ||y_i - B c||^2 per curve, then Lloyd k-means on the coefficient vectors

    Parameters
    ----------
    Y : array-like, shape (n, T)
        One curve per row on a common grid.
    K : int
        Number of clusters.
    basis : array-like, shape (T, p), optional
        B-spline basis on the grid; raw curves if omitted.
    iters : int
        Fixed number of Lloyd iterations.

    Returns
    -------
    RichResult
        ``labels``, ``centers``, ``coef``, ``wss``, ``K``, ``n``.

    References
    ----------
    Abraham, Cornillon, Matzner-Lober and Molinari (2003), Unsupervised
    curve clustering using B-splines, Scandinavian Journal of Statistics
    30(3):581-595.  The article itself is behind a paywall and could not
    be obtained; the two-stage form implemented here -- fit B-spline
    coefficients per curve, then k-means on the coefficient vectors --
    is as the method is described in the functional-clustering review
    literature (arXiv:1803.00276).
    """
    Y = C.mat(Y)
    n = len(Y); K = int(K)
    if n < K or K < 1:
        raise ValueError("need at least K curves")
    if basis is None:
        coef = [list(r) for r in Y]
    else:
        B = C.mat(basis)
        coef = [C.lstsq(B, list(row))[0] for row in Y]
    p = len(coef[0])
    order = sorted(range(n), key=lambda i: (coef[i][0], i))
    centers = [list(coef[order[(j * n) // K]]) for j in range(K)]
    labels = [0] * n
    for _ in range(int(iters)):
        for i in range(n):
            best, bj = float("inf"), 0
            for j in range(K):
                d = sum((coef[i][k] - centers[j][k]) ** 2 for k in range(p))
                if d < best:
                    best, bj = d, j
            labels[i] = bj
        for j in range(K):
            mem = [coef[i] for i in range(n) if labels[i] == j]
            if mem:
                centers[j] = [sum(m[k] for m in mem) / len(mem) for k in range(p)]
    wss = sum(sum((coef[i][k] - centers[labels[i]][k]) ** 2 for k in range(p))
              for i in range(n))
    return RichResult(payload={
        "labels": labels, "centers": centers, "coef": coef, "wss": wss,
        "K": K, "n": n, "method": "Functional clustering (k-means on B-spline coefficients)"})


functional_clustering = fdaclust


def cheatsheet():
    return "fclust: Functional clustering by k-means on B-spline coefficients."
