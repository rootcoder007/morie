# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Partial least squares regression (ESL Ch 3.5.2)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_pls"]


def esl_pls(X, y, M):
    """
    PLS: directions chosen to maximise Cov(z_m, y).

    Algorithm 3.3 in ESL: at each step take phi_m with entries
    <x_j, y>, form z_m = sum phi_mj x_j, regress y on z_m, then
    ORTHOGONALISE the remaining predictors against z_m. Unlike PCR
    (eslpcr), the response drives the directions, so PLS can pick up
    a low-variance direction that predicts well -- and with M = p
    both methods collapse back to OLS, which the doctest checks.
    Coefficients are on the centred scale, intercept separate.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix WITHOUT an intercept column.
    y : array-like, shape (n,)
        Response.
    M : int
        Directions to extract, 1 <= M <= min(n - 1, p).

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, intercept,
        y_variance_explained, M, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 3.5.2 (Alg. 3.3).

    Examples
    --------
    >>> import numpy as np
    >>> X = [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]]
    >>> y = [1.0, 2.0, 5.0, 5.0]
    >>> full = esl_pls(X, y, 2)
    >>> Xc = np.asarray(X) - np.mean(X, axis=0)
    >>> ols = np.linalg.lstsq(Xc, np.asarray(y) - np.mean(y), rcond=None)[0]
    >>> bool(np.allclose(full["beta"], ols))
    True
    >>> esl_pls(X, y, 1)["y_variance_explained"] > 0.5
    True
    >>> esl_pls(X, y, 0)
    Traceback (most recent call last):
        ...
    ValueError: M must lie in [1, 2]; got 0.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    M = int(M)
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    kmax = min(n - 1, p)
    if not 1 <= M <= kmax:
        raise ValueError(f"M must lie in [1, {kmax}]; got {M}.")
    xbar = X.mean(axis=0)
    ybar = float(y.mean())
    Xw = (X - xbar).copy()
    yc = y - ybar
    # NIPALS bookkeeping: weights W, X-loadings P, y-loadings q. The
    # coefficients must be mapped back through the deflation with
    # B = W (P'W)^-1 q -- accumulating theta * phi directly is wrong
    # because after the first step phi lives in the DEFLATED space.
    W, P, q = [], [], []
    fit = np.zeros(n)
    for m in range(M):
        phi = Xw.T @ yc
        nrm = float(np.linalg.norm(phi))
        if nrm <= 0:
            break
        phi = phi / nrm
        z = Xw @ phi
        zz = float(z @ z)
        if zz <= 0:
            break
        theta = float(z @ yc) / zz
        load = (Xw.T @ z) / zz
        W.append(phi); P.append(load); q.append(theta)
        fit = fit + theta * z
        Xw = Xw - np.outer(z, load)
    if W:
        Wm = np.column_stack(W); Pm = np.column_stack(P)
        qv = np.asarray(q, dtype=float)
        beta = Wm @ np.linalg.solve(Pm.T @ Wm, qv)
    else:
        beta = np.zeros(p)
    T = W
    tss = float(yc @ yc)
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "intercept": ybar - float(xbar @ beta),
        "y_variance_explained": float((fit @ fit) / tss) if tss > 0 else float("nan"),
        "M": len(T), "n": int(n), "p": int(p),
        "method": "PLS (ESL Alg. 3.3): directions maximise Cov(z, y), deflate X"})


def cheatsheet():
    return "eslpls: y-driven directions + deflation; M=p collapses to OLS"
