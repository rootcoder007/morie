# morie.fn -- function file (rootcoder007/morie)
"""Liu simplicial depth."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["simplicial_depth"]


def simplicial_depth(X, theta):
    """
    Liu simplicial depth

    Formula: P(theta in the simplex of d+1 random points)

    The sample version counts the simplices spanned by d+1 data points
    that contain theta, over all C(n, d+1) of them.  In one dimension a
    simplex is an interval, so the depth is the fraction of pairs
    straddling theta and its population value is exactly 2 F(theta)
    (1 - F(theta)) -- maximised at the median, which is the closed form
    used to check this.

    Parameters
    ----------
    X : array-like
        n x d data matrix, d = 1 or 2.
    theta : array-like
        The point, length d.

    Returns
    -------
    result : dict
        Keys: estimate (depth), depth, n_containing, n_simplices,
        ecdf, closed_form_1d, n, d.

    References
    ----------
    Liu (1990), On a notion of data depth based on random simplices,
    Ann. Statist. 18(1):405-414.
    """
    M = core.mat(X)
    n = len(M)
    if n < 2:
        raise ValueError("need at least two data points")
    d = len(M[0])
    th = core.vec(theta)
    if len(th) != d:
        raise ValueError("X and theta must have the same dimension")
    if d not in (1, 2):
        raise ValueError("simplicial depth here supports d = 1 or 2")
    cnt = 0
    tot = 0
    if d == 1:
        xs = [r[0] for r in M]
        for i in range(n):
            for j in range(i + 1, n):
                tot += 1
                lo = min(xs[i], xs[j])
                hi = max(xs[i], xs[j])
                if lo <= th[0] <= hi:
                    cnt += 1
        F = sum(1 for v in xs if v <= th[0]) / float(n)
        cf = 2.0 * F * (1.0 - F)
    else:
        if n < 3:
            raise ValueError("need at least three points in two dimensions")

        def side(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    tot += 1
                    a, b, c = M[i], M[j], M[k]
                    s1 = side(a, b, th)
                    s2 = side(b, c, th)
                    s3 = side(c, a, th)
                    if (s1 >= 0 and s2 >= 0 and s3 >= 0) or \
                       (s1 <= 0 and s2 <= 0 and s3 <= 0):
                        cnt += 1
        F = float("nan")
        cf = float("nan")
    depth = cnt / float(tot) if tot else float("nan")
    return RichResult(payload={
        "estimate": depth,
        "depth": depth,
        "n_containing": cnt,
        "n_simplices": tot,
        "ecdf": F,
        "closed_form_1d": cf,
        "n": n,
        "d": d,
        "method": "Liu simplicial depth",
    })


def cheatsheet():
    return "depthS: Liu simplicial depth"


# compact alias per ledger/NAMING.md
simplicialdepth = simplicial_depth
