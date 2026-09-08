# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Logistic regression by Newton-Raphson (IRLS)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_logistic_regression"]


def wasserman_logistic_regression(X, y, max_iter=100, tol=1e-10):
    """
    Logistic regression MLE.

    Formula: P(Y = 1 | X) = exp(X beta) / (1 + exp(X beta)), fit by
    Newton-Raphson (equivalently IRLS): beta <- beta +
    (X'WX)^{-1} X'(y - p), W = diag(p(1-p)). Standard errors are
    sqrt(diag((X'WX)^{-1})) at convergence. Perfect separation makes
    the MLE infinite — detected via exploding coefficients and
    refused with a clear message.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix (add your own intercept).
    y : array-like, shape (n,)
        Binary response in {0, 1}.
    max_iter, tol
        Newton controls.

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, se,
        log_likelihood, iterations, converged, n, p, method.

    References
    ----------
    Wasserman (2004), Ch 13, section 13.7.

    Examples
    --------
    Intercept-only model recovers logit of the base rate:

    >>> import math
    >>> X = [[1.0]] * 4
    >>> out = wasserman_logistic_regression(X, [1, 1, 1, 0])
    >>> abs(out["beta"][0] - math.log(3.0)) < 1e-8
    True
    >>> round(out["log_likelihood"], 12) == round(3 * math.log(0.75) + math.log(0.25), 12)
    True
    >>> Xs = [[1.0, -1.0], [1.0, -2.0], [1.0, 1.0], [1.0, 2.0]]
    >>> wasserman_logistic_regression(Xs, [0, 0, 1, 1])
    Traceback (most recent call last):
        ...
    ValueError: perfect separation: the MLE is infinite; regularise or change the model.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("the response must be binary 0/1.")
    beta = np.zeros(p)
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1.0 - mu)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            raise ValueError("perfect separation: the MLE is infinite; regularise or change the model.")
        beta = beta + step
        if np.max(np.abs(beta)) > 30.0:
            raise ValueError("perfect separation: the MLE is infinite; regularise or change the model.")
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    ll = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    W = mu * (1.0 - mu)
    cov = np.linalg.inv(X.T @ (X * W[:, None]))
    se = np.sqrt(np.diag(cov))
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "se": [float(v) for v in se], "log_likelihood": ll,
        "iterations": int(it), "converged": bool(converged),
        "n": int(n), "p": int(p),
        "method": "logistic MLE by Newton-Raphson; separation refused"})


def cheatsheet():
    return "wsmlgr: IRLS beta += (X'WX)^-1 X'(y-p); |beta|>30 -> separation error"
