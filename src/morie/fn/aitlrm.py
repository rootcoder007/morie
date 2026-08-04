# morie.fn -- function file (rootcoder007/morie)
"""Log-ratio mean of a compositional data set."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['complrmean', 'compositional_lrmean']


def complrmean(X, total=1.0):
    """Log-ratio mean of a compositional data set.

    Formula: lrmean(X) = clr^-1( (1/n) sum_r clr(x_r) ) = C( g_1, ..., g_D ), g_i the geometric mean of column i

    Parameters
    ----------
    X : array-like, shape (n, D)
        One composition per row; all parts strictly positive.
    total : float
        Constant kappa the closure sums to.

    Returns
    -------
    RichResult
        ``mean``, ``clr_mean``, ``geometric_mean``, ``total``, ``n``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read; see EXTERNAL_SOURCES.md.  The arithmetic mean of compositions is not a compositional statistic; the mean taken in log-ratio coordinates and mapped back is.  Averaging clr coordinates and inverting gives exactly the closed vector of per-part geometric means, so this is the same estimate as the compositional centre in morie.fn.aitcen -- both are returned so the identity is visible rather than assumed.  Implemented in the standard published form.  The log-ratio algebra it rests on was verified against Mateu-Figueras, Pawlowsky-Glahn and Egozcue, arXiv:0802.2643 Sect. 4.1 (fetched and archived), but this particular definition is not printed there and could not be checked against Aitchison's own text.
    """
    Xm = C.mat(X)
    n = len(Xm)
    if n == 0:
        raise ValueError("X must have at least one composition")
    D = len(Xm[0])
    for row in Xm:
        if any(v <= 0.0 for v in row):
            raise ValueError("compositions must be strictly positive")
    zbar = [0.0] * D
    for row in Xm:
        lg = sum(math.log(v) for v in row) / D
        for j in range(D):
            zbar[j] += (math.log(row[j]) - lg) / n
    m = max(zbar)
    e = [math.exp(v - m) for v in zbar]
    s = sum(e)
    k = float(total)
    gm = [math.exp(sum(math.log(Xm[r][j]) for r in range(n)) / n) for j in range(D)]
    return RichResult(payload={
        "mean": [k * v / s for v in e], "clr_mean": zbar,
        "geometric_mean": gm, "total": k, "n": n, "D": D,
        "method": "Compositional log-ratio mean"})


compositional_lrmean = complrmean


def cheatsheet():
    return 'aitlrm: Log-ratio mean of a compositional data set.'
