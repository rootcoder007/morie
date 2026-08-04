# morie.fn -- function file (rootcoder007/morie)
"""Binary cross-entropy (logistic) loss."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['bcelo', 'binary_crossentropy_loss']


def bcelo(Y, P):
    """Binary cross-entropy (logistic) loss.

    Formula: L(w) = -sum_i sum_j [ y_ij log(yhat_ij) + (1 - y_ij) log(1 - yhat_ij) ]

    Parameters
    ----------
    Y : array-like, shape (n, L)
        Observed 0/1 outcomes; a flat vector is read as one column.
    P : array-like, shape (n, L)
        Predicted success probabilities, strictly inside (0, 1).

    Returns
    -------
    RichResult
        ``loss``, ``mean_loss``, ``n``, ``L``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 10, Sect. 10.7, pp. 400-403.  Read from the chapter PDF, not recalled.  BOOK DEFECT: the display in Sect. 10.7.2 prints this loss without its leading minus sign, even though the same paragraph calls it the negative log-likelihood of a product of Bernoulli distributions and the Poisson loss two displays later does carry its sign.  What is implemented here is the quantity the surrounding text requires -- a loss that is minimised -- not the sign-dropped display.  The book has not been silently corrected elsewhere.
    """
    Y = C.mat(Y); P = C.mat(P)
    n = len(Y)
    if n == 0 or n != len(P) or len(Y[0]) != len(P[0]):
        raise ValueError("Y and P must be non-empty and the same shape")
    L = len(Y[0])
    loss = 0.0
    for i in range(n):
        for j in range(L):
            p = P[i][j]
            if not 0.0 < p < 1.0:
                raise ValueError("predicted probabilities must lie strictly in (0, 1)")
            loss -= Y[i][j] * math.log(p) + (1.0 - Y[i][j]) * math.log(1.0 - p)
    return RichResult(payload={
        "loss": loss, "mean_loss": loss / (n * L), "n": n, "L": L,
        "method": "Binary cross-entropy loss, MVSML Sect. 10.7.2"})


binary_crossentropy_loss = bcelo


def cheatsheet():
    return 'bcelO: Binary cross-entropy (logistic) loss.'
