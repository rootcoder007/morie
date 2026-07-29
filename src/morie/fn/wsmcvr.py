# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""K-fold cross-validation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_kfold_cv"]


def wasserman_kfold_cv(X, y, model, k):
    """
    K-fold cross-validated prediction error.

    Formula: CV = (1/n) sum_i (Y_i - m_hat^{(-fold(i))}(X_i))^2.
    Folds are CONTIGUOUS blocks in the given order (deterministic —
    shuffle beforehand if the order is informative; no hidden RNG).
    ``model`` is a callable (X_train, y_train, X_test) -> predictions;
    None means OLS on the given design.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    model : callable or None
        Fit-and-predict function; None = least squares.
    k : int
        Number of folds, 2 <= k <= n.

    Returns
    -------
    result : dict
        Keys: estimate (CV mean squared error), fold_mse, fold_sizes,
        k, n, method.

    References
    ----------
    Wasserman (2004), Ch 13, section 13.6 (cross-validation).

    Examples
    --------
    A perfect linear relation cross-validates to ~zero error:

    >>> X = [[1.0, float(i)] for i in range(8)]
    >>> y = [1.0 + 2.0 * i for i in range(8)]
    >>> out = wasserman_kfold_cv(X, y, None, 4)
    >>> round(out["estimate"], 10)
    0.0
    >>> out["fold_sizes"]
    [2, 2, 2, 2]
    >>> wasserman_kfold_cv(X, y, None, 1)
    Traceback (most recent call last):
        ...
    ValueError: cross-validation needs 2 <= k <= n; got k=1, n=8.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = X.shape[0]
    k = int(k)
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    if not 2 <= k <= n:
        raise ValueError(f"cross-validation needs 2 <= k <= n; got k={k}, n={n}.")
    if model is None:
        def model(Xtr, ytr, Xte):
            beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
            return Xte @ beta
    bounds = np.linspace(0, n, k + 1).astype(int)
    fold_mse, fold_sizes, sq_sum = [], [], 0.0
    for f in range(k):
        lo, hi = bounds[f], bounds[f + 1]
        te = np.arange(lo, hi)
        tr = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        pred = np.asarray(model(X[tr], y[tr], X[te]), dtype=float)
        sq = (y[te] - pred) ** 2
        fold_mse.append(float(np.mean(sq)))
        fold_sizes.append(int(te.size))
        sq_sum += float(np.sum(sq))
    return RichResult(payload={
        "estimate": float(sq_sum / n), "fold_mse": fold_mse,
        "fold_sizes": fold_sizes, "k": k, "n": int(n),
        "method": "k-fold CV, contiguous deterministic folds, squared error"})


def cheatsheet():
    return "wsmcvr: contiguous folds, no hidden RNG; CV = total squared error / n"
