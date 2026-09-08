# morie.fn -- function file (rootcoder007/morie)
"""Centre of a compositional data set."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['compcen', 'aitchison_center']


def compcen(X, total=1.0):
    """Centre of a compositional data set.

    The arithmetic mean of compositions is not a sensible centre -- Aitchison shows a case where it sits outside the data cloud entirely, looking more like an outlier than a centre. The geometric mean, closed back to the simplex, is the estimate that respects the geometry.


    Formula: xi-hat = C(g_1, ..., g_D), g_i the geometric mean of the ith component

    Parameters
    ----------
    X : array-like, shape (n, D)
        One composition per row; all entries strictly positive.
    total : float
        Constant the closure sums to.

    Returns
    -------
    RichResult
        ``center``, ``geometric_mean``, ``n``, ``D``.

    References
    ----------
    Aitchison, A Concise Guide to Compositional Data Analysis,
    Chapter 2.  Verified against the text: the centre estimate is
    xi-hat = C(g_1, ..., g_D) with g_i the geometric mean of the ith
    component, and totvar(x) = trace(Gamma) = (1/D) sum_{i<j}
    var{log(x_i/x_j)}.
    """
    X = C.mat(X)
    n = len(X); D = len(X[0])
    for row in X:
        if any(v <= 0 for v in row):
            raise ValueError("compositions must be strictly positive")
    g = [math.exp(sum(math.log(X[i][j]) for i in range(n)) / n) for j in range(D)]
    s = sum(g)
    return RichResult(payload={
        "center": [float(total) * v / s for v in g], "geometric_mean": g,
        "n": n, "D": D, "method": "Compositional centre (closed geometric mean)"})


aitchison_center = compcen


def cheatsheet():
    return "aitcen: Centre of a compositional data set."
