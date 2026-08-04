# morie.fn -- function file (rootcoder007/morie)
"""Variation matrix of a compositional data set."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['compvarmat', 'aitchison_variation']


def compvarmat(X):
    """Variation matrix of a compositional data set.

    Formula: tau_ij = var( log(x_i / x_j) ) over the rows;  totvar = (1/(2D)) sum_i sum_j tau_ij

    Parameters
    ----------
    X : array-like, shape (n, D)
        One composition per row; all parts strictly positive.

    Returns
    -------
    RichResult
        ``variation``, ``total_variation``, ``n``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read; see EXTERNAL_SOURCES.md.  The variation matrix collects the sample variances of every pairwise log-ratio; it is symmetric with a zero diagonal, and its off-diagonal entries are the only compositional statistics that do not change when parts are dropped.  The total variation reported here is (1/(2D)) sum over the full matrix, equivalently (1/D) sum over i < j, which is the normalisation already in use elsewhere in this shelf (morie.fn.aitcen, morie.fn.aittvr).  Variances use the n - 1 divisor in both language arms.  Implemented in the standard published form.  The log-ratio algebra it rests on was verified against Mateu-Figueras, Pawlowsky-Glahn and Egozcue, arXiv:0802.2643 Sect. 4.1 (fetched and archived), but this particular definition is not printed there and could not be checked against Aitchison's own text.
    """
    Xm = C.mat(X)
    n = len(Xm)
    if n < 2:
        raise ValueError("the variation matrix needs at least two compositions")
    D = len(Xm[0])
    for row in Xm:
        if any(v <= 0.0 for v in row):
            raise ValueError("compositions must be strictly positive")
    L = [[math.log(v) for v in row] for row in Xm]
    T = [[0.0] * D for _ in range(D)]
    for i in range(D):
        for j in range(i + 1, D):
            d = [L[r][i] - L[r][j] for r in range(n)]
            m = sum(d) / n
            s = sum((v - m) ** 2 for v in d) / (n - 1)
            T[i][j] = s
            T[j][i] = s
    tot = sum(T[i][j] for i in range(D) for j in range(D)) / (2.0 * D)
    return RichResult(payload={
        "variation": T, "total_variation": tot, "n": n, "D": D,
        "method": "Compositional variation matrix"})


aitchison_variation = compvarmat


def cheatsheet():
    return 'aitvar: Variation matrix of a compositional data set.'
