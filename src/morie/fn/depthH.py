# morie.fn -- function file (rootcoder007/morie)
"""Tukey halfspace depth."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["halfspace_depth"]


def halfspace_depth(X, theta):
    """Depth of a point as the thinnest halfspace through it.

    Depth is what a median means once there is more than one dimension:
    the deepest point is the one no halfspace can isolate.  The
    definition is affine invariant, which is why it survives a change of
    units that would wreck a coordinatewise median.

    In the plane the minimum over all directions is attained at a
    direction normal to one of the vectors from ``theta`` to a data
    point, so enumerating those gives the exact depth.  Above two
    dimensions no such finite exact set is cheap, and the minimum is
    taken over the directions to the data points themselves -- this is
    the Rousseeuw-Struyf direction scheme and it returns an UPPER bound
    on the true depth, reported as ``exact = False``.

    Formula: ``depth(theta) = min_u #{i : u' (x_i - theta) >= 0} / n``.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Data cloud.
    theta : array-like, shape (p,)
        Point whose depth is wanted.

    Returns
    -------
    RichResult
        ``estimate`` (depth as a proportion), ``count`` (the minimising
        halfspace count), ``exact`` (1 in the plane, 0 above it),
        ``n``, ``p``.

    References
    ----------
    Tukey, J. W. (1975).  Mathematics and the picturing of data.
    Proceedings of the International Congress of Mathematicians,
    Vancouver, 2:523-531.  The planar algorithm is Rousseeuw, P. J. &
    Ruts, I. (1996), Algorithm AS 307: bivariate location depth,
    Applied Statistics 45:516-526; the direction scheme above two
    dimensions is Rousseeuw, P. J. & Struyf, A. (1998), Computing
    location depth and regression depth in higher dimensions,
    Statistics and Computing 8:193-203.
    """
    Xm = C.mat(X)
    t = C.vec(theta)
    n, p = C.shape(Xm)
    d = [[Xm[i][j] - t[j] for j in range(p)] for i in range(n)]
    dirs = []
    if p == 2:
        for row in d:
            if row[0] == 0.0 and row[1] == 0.0:
                continue
            dirs.append([-row[1], row[0]])
            dirs.append([row[1], -row[0]])
        exact = 1
    else:
        for row in d:
            if all(v == 0.0 for v in row):
                continue
            dirs.append(list(row))
            dirs.append([-v for v in row])
        exact = 0
    if not dirs:
        return RichResult(payload={
            "estimate": 1.0, "count": n, "exact": exact, "n": n, "p": p,
            "method": "Tukey halfspace depth"})
    best = n
    for u in dirs:
        cnt = sum(1 for row in d if sum(u[j] * row[j] for j in range(p)) >= 0.0)
        if cnt < best:
            best = cnt
    return RichResult(payload={
        "estimate": best / n, "count": best, "exact": exact, "n": n, "p": p,
        "method": "Tukey halfspace depth"})


halfspacedepth = halfspace_depth


def cheatsheet():
    return "depthH: Tukey halfspace depth."
