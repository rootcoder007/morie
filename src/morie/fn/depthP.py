# morie.fn -- function file (rootcoder007/morie)
"""Stahel-Donoho projection depth."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["projection_depth"]


def projection_depth(x, X, n_dir=180):
    """
    Stahel-Donoho projection depth

    Formula: d = 1 / (1 + sup_u |u'x - Med(u'X)| / MAD(u'X))

    The outlyingness of a point is the worst standardised deviation
    over all one-dimensional projections, and the depth is its
    reciprocal transform, so depth lies in (0, 1] and equals exactly 1
    at a point whose projection is the median in every direction.  In
    one dimension the supremum is attained at u = +/-1, so the depth is
    available in closed form.

    Parameters
    ----------
    x : array-like
        The point, length d.
    X : array-like
        n x d data matrix.
    n_dir : int
        Number of equally spaced directions used when d = 2; for d = 1
        the two signed directions are exact.

    Returns
    -------
    result : dict
        Keys: estimate (depth), depth, outlyingness, med, mad,
        worst_dir, n, d.

    References
    ----------
    Stahel (1981), PhD thesis, ETH Zurich.
    Donoho (1982), PhD qualifying paper, Harvard University.
    Zuo & Serfling (2000), Ann. Statist. 28(2):461-482.
    """
    p = core.vec(x)
    M = core.mat(X)
    n = len(M)
    if n < 2:
        raise ValueError("need at least two data points")
    d = len(M[0])
    if len(p) != d:
        raise ValueError("x and X must have the same dimension")
    if d == 1:
        dirs = [[1.0]]
    elif d == 2:
        nd = int(n_dir)
        if nd < 2:
            raise ValueError("n_dir must be at least 2")
        dirs = [[math.cos(math.pi * t / nd), math.sin(math.pi * t / nd)]
                for t in range(nd)]
    else:
        raise ValueError("projection depth here supports d = 1 or 2")
    worst = 0.0
    wmed = float("nan")
    wmad = float("nan")
    wdir = 0
    for q, u in enumerate(dirs):
        proj = [sum(M[i][t] * u[t] for t in range(d)) for i in range(n)]
        pu = sum(p[t] * u[t] for t in range(d))
        med = core.median(proj)
        mad = core.mad(proj)
        if mad <= 0.0:
            continue
        o = abs(pu - med) / mad
        if o > worst:
            worst = o
            wmed = med
            wmad = mad
            wdir = q
    depth = 1.0 / (1.0 + worst)
    return RichResult(payload={
        "estimate": depth,
        "depth": depth,
        "outlyingness": worst,
        "med": wmed,
        "mad": wmad,
        "worst_dir": wdir,
        "n": n,
        "d": d,
        "method": "Stahel-Donoho projection depth",
    })


def cheatsheet():
    return "depthP: Stahel-Donoho projection depth"


# compact alias per ledger/NAMING.md
projectiondepth = projection_depth
