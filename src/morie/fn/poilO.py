# morie.fn -- function file (rootcoder007/morie)
"""Poisson loss for count outcomes."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['poislo', 'poisson_loss_dnn', 'poissonlossdnn']


def poislo(Y, Yhat):
    """Poisson loss for count outcomes.

    Formula: L(w) = sum_i sum_j [ yhat_ij - y_ij log(yhat_ij) ]

    Parameters
    ----------
    Y : array-like, shape (n, L)
        Observed counts; a flat vector is read as one column.
    Yhat : array-like, shape (n, L)
        Predicted Poisson means, strictly positive.

    Returns
    -------
    RichResult
        ``loss``, ``mean_loss``, ``n``, ``L``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 10, Sect. 10.7, pp. 400-403.  Read from the chapter PDF, not recalled.  The Poisson display carries its sign correctly and is implemented exactly as printed.
    """
    Y = C.mat(Y); H = C.mat(Yhat)
    n = len(Y)
    if n == 0 or n != len(H) or len(Y[0]) != len(H[0]):
        raise ValueError("Y and Yhat must be non-empty and the same shape")
    L = len(Y[0])
    loss = 0.0
    for i in range(n):
        for j in range(L):
            if H[i][j] <= 0.0:
                raise ValueError("predicted Poisson means must be strictly positive")
            loss += H[i][j] - Y[i][j] * math.log(H[i][j])
    return RichResult(payload={
        "loss": loss, "mean_loss": loss / (n * L), "n": n, "L": L,
        "method": "Poisson loss, MVSML Sect. 10.7.2"})


poisson_loss_dnn = poislo
poissonlossdnn = poislo


def cheatsheet():
    return 'poilO: Poisson loss for count outcomes.'
