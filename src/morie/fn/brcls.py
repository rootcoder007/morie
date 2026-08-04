# morie.fn -- function file (rootcoder007/morie)
"""Brier score for a categorical predictive distribution."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['brierscore', 'brier_score']


def brierscore(P, y):
    """Brier score for a categorical predictive distribution.

    Formula: BS = (1/T) sum_i sum_c (pihat_ic - d_ic)^2,  d_ic = 1 iff y_i = c

    Parameters
    ----------
    P : array-like, shape (T, C)
        Predicted category probabilities, one row per observation.
    y : array-like
        Observed category index, 1-based, length T.

    Returns
    -------
    RichResult
        ``brier``, ``brier_scaled``, ``n``, ``C``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 4, Eq. (4.14) p. 136.  The book notes the categorical Brier score ranges over [0, 2] and suggests reporting BS/2; ``brier_scaled`` is that halved value.  Read from the chapter PDF, not recalled.
    """
    P = C.mat(P)
    y = [int(v) for v in y]
    T = len(P)
    if T == 0:
        raise ValueError("p must have at least one row")
    K = len(P[0])
    if len(y) != T:
        raise ValueError("y must have one entry per row of p")
    tot = 0.0
    for i in range(T):
        if not 1 <= y[i] <= K:
            raise ValueError("y must be a 1-based category index")
        for c in range(K):
            d = 1.0 if y[i] == c + 1 else 0.0
            tot += (P[i][c] - d) ** 2
    bs = tot / T
    return RichResult(payload={
        "brier": bs, "brier_scaled": bs / 2.0, "n": T, "C": K,
        "method": "Brier score, MVSML Eq. (4.14)"})


brier_score = brierscore


def cheatsheet():
    return 'brcls: Brier score for a categorical predictive distribution.'
