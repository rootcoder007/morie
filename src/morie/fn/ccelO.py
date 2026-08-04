# morie.fn -- function file (rootcoder007/morie)
"""Categorical cross-entropy loss."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['ccelo', 'categorical_crossentropy_loss']


def ccelo(Y, P):
    """Categorical cross-entropy loss.

    Formula: L(w) = -sum_i sum_c y_ic log(yhat_ic)

    Parameters
    ----------
    Y : array-like, shape (n, C)
        Observed class-indicator matrix, one row per record.
    P : array-like, shape (n, C)
        Predicted class probabilities, strictly positive.

    Returns
    -------
    RichResult
        ``loss``, ``mean_loss``, ``n``, ``C``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 10, Sect. 10.7, pp. 400-403.  Read from the chapter PDF, not recalled.  BOOK DEFECT: the display in Sect. 10.7.2 prints this loss without its leading minus sign, even though the same paragraph calls it the negative log-likelihood of a product of Bernoulli distributions and the Poisson loss two displays later does carry its sign.  What is implemented here is the quantity the surrounding text requires -- a loss that is minimised -- not the sign-dropped display.  The book has not been silently corrected elsewhere.
    """
    Y = C.mat(Y); P = C.mat(P)
    n = len(Y)
    if n == 0 or n != len(P) or len(Y[0]) != len(P[0]):
        raise ValueError("Y and P must be non-empty and the same shape")
    K = len(Y[0])
    loss = 0.0
    for i in range(n):
        for j in range(K):
            if P[i][j] <= 0.0:
                raise ValueError("predicted probabilities must be strictly positive")
            loss -= Y[i][j] * math.log(P[i][j])
    return RichResult(payload={
        "loss": loss, "mean_loss": loss / n, "n": n, "C": K,
        "method": "Categorical cross-entropy loss, MVSML Sect. 10.7.2"})


categorical_crossentropy_loss = ccelo


def cheatsheet():
    return 'ccelO: Categorical cross-entropy loss.'
