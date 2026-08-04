# morie.fn -- function file (rootcoder007/morie)
"""Geometric median of a composition in clr coordinates."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["clrmedian", "compositional_median"]


def clrmedian(X, steps=100, eps=1e-12):
    """Spatial median of compositional data in centred log-ratio space.

    Compositions carry only relative information, so location has to be
    estimated in log-ratio coordinates.  The centred log-ratio transform
    of a composition x on the D-part simplex is

        clr(x)_j = log( x_j / (prod_l x_l)^(1/D) ),

    an isometry onto the hyperplane sum_j y_j = 0.  The spatial (L1)
    median there minimises sum_i || clr(x_i) - m ||, and is found by the
    Weiszfeld iteration

        m <- sum_i clr(x_i)/d_i  /  sum_i 1/d_i,   d_i = ||clr(x_i) - m||,

    with points sitting on the current iterate skipped.  The estimate is
    returned both in clr coordinates and back on the simplex, by the
    inverse clr (exponentiate, then close to unit sum).

    Parameters
    ----------
    X : array-like, shape (n, D)
        Strictly positive compositions, one per row; rows need not be
        closed, since clr is scale invariant.
    steps : int
        Fixed Weiszfeld iteration count; no tolerance early exit.
    eps : float
        Distances below this are treated as coincident and skipped.

    Returns
    -------
    RichResult
        ``median``, ``clrmed``, ``objective``, ``clrmean``, ``n``,
        ``D``, ``steps``.

    References
    ----------
    Filzmoser, P. and Hron, K. (2008), "Outlier detection for
    compositional data using robust methods", Mathematical Geosciences
    40(3), 233-248, which estimates location for compositions in
    log-ratio coordinates; the clr transform is Aitchison's, and the
    spatial median is minimised by Weiszfeld's algorithm (Weiszfeld 1937).
    Standard published form: the Mathematical Geosciences article is
    paywalled and was not read, so only the construction the docstring
    states -- clr, then spatial median, then inverse clr -- is claimed.
    """
    M = C.mat(X)
    n, D = len(M), len(M[0])
    if n == 0 or D < 2:
        raise ValueError("need at least one composition with two parts")
    if any(v <= 0.0 for r in M for v in r):
        raise ValueError("compositions must be strictly positive")
    Y = []
    for r in M:
        lg = [math.log(v) for v in r]
        gm = sum(lg) / D
        Y.append([v - gm for v in lg])
    m = [sum(Y[i][j] for i in range(n)) / n for j in range(D)]
    cmean = list(m)
    for _ in range(int(steps)):
        num = [0.0] * D
        den = 0.0
        for i in range(n):
            d = math.sqrt(sum((Y[i][j] - m[j]) ** 2 for j in range(D)))
            if d < eps:
                continue
            for j in range(D):
                num[j] += Y[i][j] / d
            den += 1.0 / d
        if den == 0.0:
            break
        m = [v / den for v in num]
    obj = sum(math.sqrt(sum((Y[i][j] - m[j]) ** 2 for j in range(D)))
              for i in range(n))
    e = [math.exp(v) for v in m]
    se = sum(e)
    return RichResult(payload={
        "median": [v / se for v in e], "clrmed": m, "objective": obj,
        "clrmean": cmean, "n": n, "D": D, "steps": int(steps),
        "method": "Spatial median in clr coordinates (Weiszfeld iteration)"})


compositional_median = clrmedian


def cheatsheet():
    return "aitcmd: Geometric median of a composition in clr coordinates."
