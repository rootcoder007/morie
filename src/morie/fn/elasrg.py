# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Elastic net in the two-penalty (lambda1, lambda2) form."""

import numpy as np

from ._richresult import RichResult
from .eslnln import _enet

__all__ = ["elastic_net_regression"]


def elastic_net_regression(y, X, lambda1, lambda2, max_iter=10000, tol=1e-12):
    """
    Elastic net stated with SEPARATE L1 and L2 penalties.

    Formula: min ||y - X beta||^2 + lambda1 ||beta||_1 +
    lambda2 ||beta||^2, the original Zou-Hastie form. Note the
    argument order is (y, X) here, not (X, y): it follows the module's
    published signature, and getting it backwards is the obvious
    mistake, so the shapes are checked and a mismatch raises rather
    than silently regressing X on y.

    This shares its solver with eslnln; the two differ only in
    parameterisation, related by lambda1 = lambda alpha and
    lambda2 = lambda (1 - alpha). The mapping is reported so results
    can be compared across the two conventions.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response (FIRST argument).
    X : array-like, shape (n, p)
        Design matrix.
    lambda1, lambda2 : float
        L1 and L2 penalties, each >= 0.
    max_iter, tol
        Coordinate-descent controls.

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, n_nonzero,
        objective, lambda1, lambda2, equivalent_lambda,
        equivalent_alpha, iterations, converged, n, p, method.

    References
    ----------
    Zou & Hastie (2005); Hastie, Tibshirani and Friedman (2009),
    Ch 3.4.3.

    Examples
    --------
    >>> X = [[1.0, 0.0], [0.0, 1.0]]
    >>> y = [3.0, -1.0]
    >>> out = elastic_net_regression(y, X, 0.5, 0.0)
    >>> [round(b, 12) for b in out["beta"]]
    [2.5, -0.5]
    >>> out["equivalent_alpha"]
    1.0
    >>> elastic_net_regression(y, X, 0.5, 0.5)["equivalent_alpha"]
    0.5
    >>> elastic_net_regression(X, y, 0.5, 0.0)
    Traceback (most recent call last):
        ...
    ValueError: y has 2 rows of width 2; pass the RESPONSE first and the design second.
    """
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim > 1 and y_arr.shape[1] > 1:
        raise ValueError(f"y has {y_arr.shape[0]} rows of width {y_arr.shape[1]}; "
                         "pass the RESPONSE first and the design second.")
    lam1 = float(lambda1)
    lam2 = float(lambda2)
    if lam1 < 0 or lam2 < 0:
        raise ValueError(f"penalties must be non-negative; got ({lam1}, {lam2}).")
    Xm, yv, beta, r, const, n, p, it, conv = _enet(X, y_arr.ravel(), lam1, lam2,
                                                   max_iter, tol)
    total = lam1 + lam2
    obj = (0.5 * float(r @ r) + lam1 * float(np.sum(np.abs(beta[~const])))
           + 0.5 * lam2 * float(np.sum(beta[~const] ** 2)))
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "n_nonzero": int(np.sum(beta != 0)), "objective": obj,
        "lambda1": lam1, "lambda2": lam2, "equivalent_lambda": total,
        "equivalent_alpha": float(lam1 / total) if total > 0 else float("nan"),
        "iterations": int(it), "converged": bool(conv), "n": int(n), "p": int(p),
        "method": "elastic net, separate (lambda1, lambda2); solver shared with eslnln"})


def cheatsheet():
    return "elasrg: (y, X) argument order; lambda1/lambda2 form, maps to eslnln's alpha"
