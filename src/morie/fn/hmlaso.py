# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lasso (L1) regression cost."""

import numpy as np

from ._richresult import RichResult
from .hml1r import geron_l1_regularization

__all__ = ["geron_lasso_cost"]

_METHOD = "Lasso cost = MSE + alpha * l1 penalty"


def geron_lasso_cost(X, y, theta, alpha, skip_bias=False):
    """
    Lasso (L1) regression cost.

    Formula: J = MSE + alpha * sum |theta_i|

    The penalty term and its subgradient are delegated to
    :func:`morie.fn.hml1r.geron_l1_regularization`; this adds the data
    term ``(1/m) sum (X theta - y)^2`` and the combined subgradient
    ``(2/m) X^T(X theta - y) + alpha*sign(theta)``.

    The cost is convex but not differentiable at any zero coefficient,
    which is exactly why lasso produces sparse solutions and why plain
    gradient descent stalls near them -- the subgradient at 0 is a set,
    and picking 0 from it is what pins a coefficient there.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix.
    y : array-like, shape (m,)
        Targets.
    theta : array-like, shape (n,)
        Coefficients.
    alpha : float
        L1 strength (non-negative).
    skip_bias : bool
        Exclude ``theta[0]`` from the penalty.

    Returns
    -------
    result : RichResult
        Keys: cost, mse, penalty, gradient, n_zero, estimate, n, method.

    Examples
    --------
    Perfect fit, so the cost is the penalty alone:
    ``alpha * |2| = 0.5 * 2 = 1``.

    >>> r = geron_lasso_cost([[1.0], [2.0]], [2.0, 4.0], [2.0], alpha=0.5)
    >>> float(r["mse"]), float(r["penalty"]), float(r["cost"])
    (0.0, 1.0, 1.0)

    With alpha = 0 the cost is exactly the MSE; residuals -2 and -4 give
    ``(4 + 16)/2 = 10``:

    >>> float(geron_lasso_cost([[1.0], [2.0]], [2.0, 4.0], [0.0], alpha=0.0)["cost"])
    10.0

    Away from the kink the subgradient matches a central difference:

    >>> X = [[1.0, 0.5], [1.0, -1.5], [1.0, 2.0]]
    >>> y = [1.0, 0.0, 3.0]
    >>> h = 1e-6
    >>> up = geron_lasso_cost(X, y, [0.4, 0.7 + h], alpha=0.3)["cost"]
    >>> dn = geron_lasso_cost(X, y, [0.4, 0.7 - h], alpha=0.3)["cost"]
    >>> g = float(geron_lasso_cost(X, y, [0.4, 0.7], alpha=0.3)["gradient"][1])
    >>> bool(abs((up - dn) / (2 * h) - g) < 1e-6)
    True

    References
    ----------
    Géron Ch 4
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_lasso_cost: X must be a non-empty 2-D array, got shape {A.shape}")
    yy = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if yy.size != A.shape[0]:
        raise ValueError(f"geron_lasso_cost: X has {A.shape[0]} rows but y has {yy.size} entries")
    t = np.atleast_1d(np.asarray(theta, dtype=float)).ravel()
    if t.size != A.shape[1]:
        raise ValueError(f"geron_lasso_cost: theta has {t.size} entries but X has {A.shape[1]} columns")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yy)):
        raise ValueError("geron_lasso_cost: X and y must be finite")

    pen = geron_l1_regularization(t, alpha, skip_bias=skip_bias)
    m = yy.size
    resid = A @ t - yy
    mse = float(np.mean(resid**2))
    cost = mse + float(pen["penalty"])
    grad = (2.0 / m) * (A.T @ resid) + pen["gradient"]

    return RichResult(
        title="Lasso cost",
        summary_lines=[("MSE", mse), ("L1 penalty", float(pen["penalty"])), ("Total cost", cost)],
        warnings=(
            ["some coefficients are exactly 0, where the L1 subgradient is a set and 0 was chosen from it."]
            if np.any(t == 0)
            else []
        ),
        interpretation="Convex but non-smooth at zero; that kink is what makes lasso select features.",
        payload={
            "cost": cost,
            "mse": mse,
            "penalty": float(pen["penalty"]),
            "gradient": grad,
            "residuals": resid,
            "n_zero": int(np.sum(t == 0)),
            "alpha": float(alpha),
            "estimate": cost,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlaso: lasso cost MSE + alpha*sum|theta| (penalty delegated to hml1r)"
