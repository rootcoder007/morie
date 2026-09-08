# morie.fn -- function file (rootcoder007/morie)
"""Local Geary's c per location."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['localgeary', 'local_gearys_c', 'localgearysc']


def localgeary(x, W, scale=True):
    """Local Geary's c per location.

    Where local Moran asks whether a location resembles its neighbours in sign, local Geary asks about squared difference, so it reacts to a location that is unlike its neighbours even when the global pattern is positive autocorrelation. x is standardised first, as the reference implementation does, using the sample standard deviation.


    Formula: c_i = sum_j w_ij (z_i - z_j)^2 with z the standardised x

    Parameters
    ----------
    x : array-like
        Values at the n locations.
    W : array-like, shape (n, n)
        Spatial weights.
    scale : bool
        Standardise x before computing c_i.

    Returns
    -------
    RichResult
        ``local``, ``global_c``, ``z``, ``n``.

    References
    ----------
    Anselin (2019), A Local Indicator of Multivariate Spatial
    Association: Extending Geary's c, Geographical Analysis 51:133-150.
    Paywalled; the univariate form c_i = sum_j w_ij (x_i - x_j)^2 and the
    standardise-first convention are as documented by spdep::localC, the
    reference implementation.
    """
    x = C.vec(x)
    W = C.mat(W)
    n = len(x)
    if scale:
        mu = sum(x) / n
        s = C.sd(x, 1)
        if s <= 0:
            raise ValueError("x has zero variance")
        z = [(v - mu) / s for v in x]
    else:
        z = list(x)
    loc = [sum(W[i][j] * (z[i] - z[j]) ** 2 for j in range(n)) for i in range(n)]
    s0 = sum(sum(row) for row in W)
    return RichResult(payload={
        "local": loc, "global_c": sum(loc) / (2.0 * s0) if s0 else float("nan"),
        "z": z, "n": n, "method": "Local Geary's c"})


local_gearys_c = localgeary
localgearysc = localgeary


def cheatsheet():
    return "gearyl: Local Geary's c per location."
