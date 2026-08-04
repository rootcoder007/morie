# morie.fn -- function file (rootcoder007/morie)
"""Maximum likelihood fit of the additive logistic-normal."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["lgtnfit", "logistic_normal_fit"]


def lgtnfit(X, ddof=1):
    """Fit the logistic-normal by transforming, then fitting a normal.

    No optimisation is needed and none is done.  The alr is a
    BIJECTION onto R^{D-1}, so the maximum-likelihood estimate on the
    simplex is exactly the multivariate-normal MLE of the transformed
    data pushed back -- an iterative fit here would be answering a
    question that has a closed form.

    The Jacobian of the transform does not depend on the parameters,
    so it shifts the log-likelihood by a constant and cannot move the
    estimate; it IS included in ``loglik`` so that value is comparable
    with ``aitlnp``.

    Formula: Y = alr(X);  muhat = mean(Y);
             Sigmahat = sum (Y_k - muhat)(Y_k - muhat)' / (n - ddof)

    Parameters
    ----------
    X : array-like, shape (n, D)
        One composition per row; strictly positive.
    ddof : int
        Divisor correction: 1 for the unbiased covariance (the
        default, matching the sibling modules), 0 for the MLE.

    Returns
    -------
    RichResult
        ``mu``, ``Sigma``, ``center`` (the fitted centre on the
        simplex), ``loglik``, ``n``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 7, in which inference for the logistic-normal class is
    carried out entirely in log-ratio coordinates because the
    transform is one-to-one and its Jacobian is free of the
    parameters.
    """
    X = C.mat(X)
    n = len(X)
    if n < 2:
        raise ValueError("at least two compositions are required")
    D = len(X[0])
    if D < 2:
        raise ValueError("a composition needs at least two parts")
    if any(len(r) != D for r in X):
        raise ValueError("every composition must have the same length")
    for r in X:
        if any(v <= 0 for v in r):
            raise ValueError("compositions must be strictly positive")
    dd = int(ddof)
    if n - dd <= 0:
        raise ValueError("not enough observations for this ddof")
    Y = [[math.log(X[k][i]) - math.log(X[k][D - 1]) for i in range(D - 1)]
         for k in range(n)]
    p = D - 1
    mu = [sum(Y[k][i] for k in range(n)) / n for i in range(p)]
    S = [[sum((Y[k][i] - mu[i]) * (Y[k][j] - mu[j]) for k in range(n))
          / (n - dd) for j in range(p)] for i in range(p)]
    e = [math.exp(v) for v in mu] + [1.0]
    s = sum(e)
    cen = [v / s for v in e]
    L = C.chol(S)
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(p))
    ll = 0.0
    for k in range(n):
        dv = [Y[k][i] - mu[i] for i in range(p)]
        q = sum(dv[i] * z for i, z in enumerate(C.solvev(S, dv)))
        ll += (-0.5 * p * math.log(2.0 * math.pi) - 0.5 * logdet
               - sum(math.log(v) for v in X[k]) - 0.5 * q)
    return RichResult(payload={
        "mu": mu, "Sigma": S, "center": cen, "loglik": ll,
        "n": float(n), "D": float(D),
        "method": "Logistic-normal MLE via the alr transform"})


logistic_normal_fit = lgtnfit


def cheatsheet():
    return "aitlnf: muhat, Sigmahat are the normal MLE of alr(X); closed form"
