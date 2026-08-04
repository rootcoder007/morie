# morie.fn -- function file (rootcoder007/morie)
"""Weiszfeld iteration for the geometric median."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["weiszfeld"]


def weiszfeld(X, tol=None, max_iter=200):
    """Iteratively reweighted mean converging on the geometric median.

    The objective is convex but not differentiable at the data points,
    and that is the whole difficulty: the update divides by the distance
    to each point, so landing exactly on one blows up.  The standard
    guard is to skip a point whose distance underflows, which is what is
    done here.  The ``tol`` argument is accepted and ignored -- a fixed
    iteration count is what makes the two language arms agree.

    Formula: ``mu <- (sum_i w_i x_i) / (sum_i w_i)`` with
    ``w_i = 1 / ||x_i - mu||``.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Points.
    tol : ignored
        Accepted for interface compatibility; see above.
    max_iter : int, default 200
        Iterations.

    Returns
    -------
    RichResult
        ``estimate`` (the median point), ``cost`` (sum of distances),
        ``n``, ``d``.

    References
    ----------
    Weiszfeld, E. (1937).  Sur le point pour lequel la somme des
    distances de n points donnes est minimum.  Tohoku Mathematical
    Journal 43:355-386.
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
    cost = sum(math.sqrt(sum((A[i][j] - mu[j]) ** 2 for j in range(d))) for i in range(n))
    return RichResult(payload={
        "estimate": mu, "cost": cost, "n": n, "d": d,
        "method": "Weiszfeld iteration, geometric median"})


def cheatsheet():
    return "weisz: Weiszfeld iteration for the geometric median."
