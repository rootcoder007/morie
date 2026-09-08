# morie.fn -- function file (rootcoder007/morie)
"""Row-normalised spatial weights."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['rownorm', 'weights_row_normalize']


def rownorm(W):
    """Row-normalised spatial weights.

    Row standardisation turns the spatial lag into a neighbour average, which is what makes the lag coefficient interpretable and bounds it by the extreme eigenvalues of the standardised matrix. Islands -- rows that sum to zero -- cannot be normalised; they are left as zeros and counted in ``islands`` rather than producing silent NaNs downstream.


    Formula: w'_ij = w_ij / sum_j w_ij

    Parameters
    ----------
    W : array-like, shape (n, n)
        Spatial weights matrix.

    Returns
    -------
    RichResult
        ``W`` (normalised), ``row_sums``, ``islands``, ``n``.

    References
    ----------
    Anselin (1988), Spatial Econometrics: Methods and Models, Kluwer.
    Not held locally; row standardisation w_ij / sum_j w_ij is the
    standard published convention and is what spdep's style = 'W'
    computes.
    """
    W = C.mat(W)
    n = len(W)
    rs = [sum(row) for row in W]
    out = [[W[i][j] / rs[i] if rs[i] != 0 else 0.0 for j in range(n)]
           for i in range(n)]
    return RichResult(payload={
        "W": out, "row_sums": rs,
        "islands": [i for i in range(n) if rs[i] == 0], "n": n,
        "method": "Row-normalised spatial weights"})


weights_row_normalize = rownorm


def cheatsheet():
    return "wmtrwn: Row-normalised spatial weights."
