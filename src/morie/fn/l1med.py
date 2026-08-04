# morie.fn -- function file (rootcoder007/morie)
"""L1 median with a spatial-rank report."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["l1_median"]


def l1_median(X, tol=None, max_iter=200):
    """Multivariate median that minimises total distance, not squared distance.

    Swapping the squared loss for the plain one buys a breakdown point
    of one half in any dimension -- half the cloud can run to infinity
    and the estimate stays put -- at the cost of the closed form.  The
    spatial rank of the solution is returned as a check: at the true L1
    median the sum of unit vectors to the data points vanishes, so a
    norm far from zero means the iteration has not arrived.

    Formula: ``min_mu sum_i ||x_i - mu||``, solved by the Weiszfeld
    iteration for a fixed number of steps.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Points.
    tol : ignored
        Accepted for interface compatibility.
    max_iter : int, default 200
        Iterations.

    Returns
    -------
    RichResult
        ``estimate`` (the L1 median), ``cost``, ``spatial_rank_norm``,
        ``n``, ``d``.

    References
    ----------
    Weiszfeld, E. (1937).  Tohoku Mathematical Journal 43:355-386.  The
    spatial-rank characterisation is Small, C. G. (1990), A survey of
    multidimensional medians, International Statistical Review
    58:263-277.
    """
    A = C.mat(X)
    n, d = C.shape(A)
    mu = [sum(A[i][j] for i in range(n)) / n for j in range(d)]
    for _ in range(int(max_iter)):
        num = [0.0] * d
        den = 0.0
        for i in range(n):
            dist = math.sqrt(sum((A[i][j] - mu[j]) ** 2 for j in range(d)))
            if dist < 1e-12:
                continue
            w = 1.0 / dist
            den += w
            for j in range(d):
                num[j] += w * A[i][j]
        if den > 0.0:
            mu = [num[j] / den for j in range(d)]
    sr = [0.0] * d
    cost = 0.0
    for i in range(n):
        dist = math.sqrt(sum((A[i][j] - mu[j]) ** 2 for j in range(d)))
        cost += dist
        if dist >= 1e-12:
            for j in range(d):
                sr[j] += (A[i][j] - mu[j]) / dist
    return RichResult(payload={
        "estimate": mu, "cost": cost,
        "spatial_rank_norm": math.sqrt(sum(v * v for v in sr)),
        "n": n, "d": d, "method": "L1 median with spatial-rank check"})


l1median = l1_median


def cheatsheet():
    return "l1med: L1 median with a spatial-rank report."
