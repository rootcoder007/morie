# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Additive model by backfitting (ESL Ch 9.1)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_gam"]


def esl_gam(X, y, g=None, max_iter=100, tol=1e-10):
    """
    Additive model g(mu) = alpha + sum_j f_j(X_j), fitted by
    backfitting.

    ESL Algorithm 9.1: initialise alpha as the mean of y and each
    f_j at zero, then cycle, refitting each f_j to the partial
    residual y - alpha - sum_{k != j} f_k and RE-CENTRING it to mean
    zero. The centring is not cosmetic — without it alpha and the f_j
    are unidentifiable, since any constant can be shifted between
    them, and the iteration wanders.

    The smoother used per coordinate is a linear fit here, which
    makes the additive model exactly equivalent to multiple linear
    regression and gives a checkable property: backfitting must
    reproduce the OLS coefficients. That equivalence is the doctest.
    ``g`` is accepted for signature compatibility and must be None or
    "identity"; a non-identity link needs local scoring
    (ESL Alg. 9.2), which this does not implement and will not fake.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Predictors, WITHOUT an intercept column.
    y : array-like, shape (n,)
        Response.
    g : None or "identity"
        Link function.
    max_iter, tol
        Backfitting controls.

    Returns
    -------
    result : dict
        Keys: estimate (intercept alpha), alpha, partial_fits
        (row-major n x p), slopes, fitted, iterations, converged,
        rss, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 9.1 (Alg. 9.1).

    Examples
    --------
    With linear smoothers, backfitting reproduces OLS exactly:

    >>> import numpy as np
    >>> X = [[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, 1.0]]
    >>> y = [1.0, 2.0, 8.0, 5.0]
    >>> out = esl_gam(X, y)
    >>> Xd = np.column_stack([np.ones(4), np.asarray(X)])
    >>> ols = np.linalg.lstsq(Xd, np.asarray(y), rcond=None)[0]
    >>> bool(np.allclose(out["slopes"], ols[1:]))
    True
    >>> abs(out["alpha"] - float(np.mean(y))) < 1e-12
    True
    >>> out["converged"]
    True
    >>> esl_gam(X, y, g="logit")
    Traceback (most recent call last):
        ...
    ValueError: only the identity link is implemented; 'logit' needs local scoring (ESL Alg. 9.2).
    """
    if g not in (None, "identity"):
        raise ValueError(f"only the identity link is implemented; '{g}' needs "
                         "local scoring (ESL Alg. 9.2).")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    alpha = float(np.mean(y))
    F = np.zeros((n, p))
    slopes = np.zeros(p)
    converged, it = False, 0
    for it in range(1, int(max_iter) + 1):
        delta = 0.0
        for j in range(p):
            partial = y - alpha - (F.sum(axis=1) - F[:, j])
            xc = X[:, j] - X[:, j].mean()
            denom = float(xc @ xc)
            b = float(xc @ (partial - partial.mean()) / denom) if denom > 0 else 0.0
            new = b * xc                      # already mean-zero
            delta = max(delta, float(np.max(np.abs(new - F[:, j]))))
            F[:, j] = new
            slopes[j] = b
        if delta < tol:
            converged = True
            break
    fitted = alpha + F.sum(axis=1)
    resid = y - fitted
    return RichResult(payload={
        "estimate": alpha, "alpha": alpha,
        "partial_fits": [float(v) for v in F.ravel()],
        "slopes": [float(v) for v in slopes],
        "fitted": [float(v) for v in fitted],
        "iterations": int(it), "converged": bool(converged),
        "rss": float(resid @ resid), "n": int(n), "p": int(p),
        "method": "backfitting (ESL Alg. 9.1), linear smoothers, components re-centred"})


def cheatsheet():
    return "eslgam: backfit partial residuals, re-centre each f_j; identity link only"
