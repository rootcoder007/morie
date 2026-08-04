# morie.fn -- function file (rootcoder007/morie)
"""Local Moran's I (LISA)."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['localmoran', 'local_morans_i']


def localmoran(x, W, mlvar=True):
    """Local Moran's I (LISA).

    A LISA decomposes a global statistic into per-observation contributions, so summing the local values and dividing by the sum of the weights returns Moran's global I. The variance divisor is n rather than n-1, matching the reference implementation's default; pass ``mlvar=False`` for the sample-variance convention.


    Formula: I_i = z_i sum_j w_ij z_j / m2, z_i = x_i - xbar, m2 = sum_i z_i^2 / n

    Parameters
    ----------
    x : array-like
        Values at the n locations.
    W : array-like, shape (n, n)
        Spatial weights.
    mlvar : bool
        Divide m2 by n (the default) rather than n-1.

    Returns
    -------
    RichResult
        ``local``, ``global_i``, ``m2``, ``z``, ``lag``, ``n``.

    References
    ----------
    Anselin (1995), Local Indicators of Spatial Association -- LISA,
    Geographical Analysis 27(2):93-115, formula (12) p.99.  The
    article is paywalled; the formula and the divide-by-n variance
    convention were taken from spdep::localmoran, the reference
    implementation, which cites that equation explicitly.
    """
    x = C.vec(x)
    W = C.mat(W)
    n = len(x)
    xbar = sum(x) / n
    z = [v - xbar for v in x]
    m2 = sum(v * v for v in z) / (n if mlvar else n - 1)
    lag = [sum(W[i][j] * z[j] for j in range(n)) for i in range(n)]
    loc = [z[i] * lag[i] / m2 for i in range(n)]
    s0 = sum(sum(row) for row in W)
    return RichResult(payload={
        "local": loc, "global_i": sum(loc) / s0 if s0 else float("nan"),
        "m2": m2, "z": z, "lag": lag, "n": n,
        "method": "Local Moran's I (LISA)"})


local_morans_i = localmoran


def cheatsheet():
    return "lismor: Local Moran's I (LISA)."
