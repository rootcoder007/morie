# morie.fn -- function file (rootcoder007/morie)
"""Closed-form ridge regression estimator."""

from . import _tail1core as C
from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['ridgesol', 'ridge_solution', 'ridgesolution']


def ridgesol(X, y, lam, add_intercept=True):
    """Closed-form ridge regression estimator.

    Formula: beta(lambda) = (X'X + lambda D)^-1 X'y,  D = diag(0, 1, ..., 1)

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix, one record per row.
    y : array-like
        Response vector of length n.
    lam : float
        Regularization parameter lambda; must be non-negative.
    add_intercept : bool
        Prepend a column of ones to X and leave its coefficient unpenalized.

    Returns
    -------
    RichResult
        ``beta``, ``fitted``, ``resid``, ``rss``, ``penalty``, ``prss``, ``lambda``, ``n``, ``p``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 3, Sect. 3.6.1 p. 82: solving grad PRSS_lambda(beta) = 0 gives beta-hat(lambda) = (X'X + lambda D)^-1 X'y with D the identity carrying a zero in its first entry.  Delegates to the chapter-3 ridge routine already verified against the book for this shelf.  Read from the chapter PDF, not recalled.
    """
    if float(lam) < 0.0:
        raise ValueError("lambda must be non-negative")
    out = G.ridge_fit(X, y, float(lam), add_intercept=bool(add_intercept))
    Xm = C.cbind1(C.mat(X)) if add_intercept else C.mat(X)
    return RichResult(payload={
        "beta": out["beta"], "fitted": out["fitted"], "resid":
            [a - b for a, b in zip(C.vec(y), out["fitted"])],
        "rss": out["rss"], "penalty": out["penalty"], "prss": out["prss"],
        "lambda": float(lam), "n": len(Xm), "p": len(Xm[0]),
        "method": "Ridge closed-form solution, MVSML Sect. 3.6.1"})


ridge_solution = ridgesol
ridgesolution = ridgesol


def cheatsheet():
    return 'ridgs: Closed-form ridge regression estimator.'
