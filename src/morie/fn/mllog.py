# morie.fn -- function file (rootcoder007/morie)
"""Maximum likelihood log-likelihood of the linear regression model."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['mlloglik', 'ml_log_likelihood_regression']


def mlloglik(X, y, beta=None, sigma2=None):
    """Maximum likelihood log-likelihood of the linear regression model.

    Formula: log L = -(n/2) log(2 pi) - n log(sigma) - (1/(2 sigma^2)) (y - X beta)'(y - X beta)

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix, one record per row.
    y : array-like
        Response vector of length n.
    beta : array-like or None
        Coefficients; None uses the OLS solution, which is also the MLE.
    sigma2 : float or None
        Error variance; None uses the MLE RSS/n.

    Returns
    -------
    RichResult
        ``loglik``, ``beta``, ``sigma2``, ``rss``, ``n``, ``p``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 3, Sect. 3.3 pp. 75-76: the likelihood of the multiple linear regression model, its logarithm as written above, and the maximum likelihood estimators -- beta-hat is the OLS solution and sigma2-hat = (1/n)(y - X beta-hat)'(y - X beta-hat), which divides by n and not by n - p.  Read from the chapter PDF, not recalled.
    """
    Xm = C.mat(X)
    y = C.vec(y)
    n = len(Xm)
    if n != len(y):
        raise ValueError("X must have one row per entry of y")
    p = len(Xm[0])
    b = C.lstsq(Xm, y)[0] if beta is None else C.vec(beta)
    if len(b) != p:
        raise ValueError("beta must have one entry per column of X")
    res = [y[i] - sum(Xm[i][j] * b[j] for j in range(p)) for i in range(n)]
    rss = sum(r * r for r in res)
    s2 = (rss / n) if sigma2 is None else float(sigma2)
    if s2 <= 0.0:
        raise ValueError("sigma2 must be positive")
    ll = (-0.5 * n * math.log(2.0 * math.pi) - 0.5 * n * math.log(s2)
          - rss / (2.0 * s2))
    return RichResult(payload={
        "loglik": ll, "beta": b, "sigma2": s2, "rss": rss, "n": n, "p": p,
        "method": "Gaussian ML log-likelihood, MVSML Sect. 3.3"})


ml_log_likelihood_regression = mlloglik


def cheatsheet():
    return 'mllog: Maximum likelihood log-likelihood of the linear regression model.'
