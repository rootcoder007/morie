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

    In the plane the count changes only when the direction crosses a
    normal to one of the vectors from ``theta`` to a data point, so it
    is constant on each arc between consecutive normals and the minimum
    is attained in the INTERIOR of an arc.  Testing the normals
    themselves is the natural-looking mistake: a data point then lies
    exactly on the boundary of the closed halfplane and is counted, so a
    point outside the convex hull comes back with positive depth instead
    of zero.  The arc midpoints are used instead, which needs no epsilon
    and is exact.  Above two dimensions no cheap exact set exists and the
    minimum is taken over the directions to the data points, which is the
    Rousseeuw-Struyf scheme and returns an UPPER bound, flagged by
    ``exact = 0``.

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
        crit = []
        for row in d:
            if row[0] == 0.0 and row[1] == 0.0:
                continue
            a = math.atan2(row[1], row[0])
            for t in (a + math.pi / 2.0, a - math.pi / 2.0):
                t = t % (2.0 * math.pi)
                crit.append(t)
        crit.sort()
        m = len(crit)
        for i in range(m):
            hi = crit[(i + 1) % m] + (2.0 * math.pi if i == m - 1 else 0.0)
            mid = 0.5 * (crit[i] + hi)
            dirs.append([math.cos(mid), math.sin(mid)])
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
