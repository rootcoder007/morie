# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ridge (L2) regression cost function."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_ridge_cost"]

_METHOD = "Ridge regression cost J(theta) = MSE + (alpha/2)||theta||^2"


def geron_ridge_cost(X, y, theta, alpha, intercept=True):
    r"""Mean squared error plus the L2 penalty.

    .. math::
        J(\boldsymbol{\theta}) = \mathrm{MSE}(\boldsymbol{\theta})
          + \frac{\alpha}{2}\sum_{i=1}^{n}\theta_i^{2}

    The sum starts at :math:`i = 1`, not 0 -- the bias is exempt.  The
    factor of one half is there so the penalty gradient is
    :math:`\alpha\theta` with no stray 2, which is how the ridge update
    ends up being plain weight decay.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    theta : array-like, shape (n,)
    alpha : float
        Non-negative penalty weight.
    intercept : bool, optional
        Treat ``theta[0]`` as the unpenalised bias.

    Returns
    -------
    RichResult
        Payload keys ``cost``, ``mse``, ``penalty``, ``gradient``,
        ``estimate`` (cost), ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-8 (Ridge Regression cost).

    Examples
    --------
    Perfect fit, so the whole cost is the penalty
    ``(alpha/2) * 3^2 = 4.5`` at ``alpha = 1``:

    >>> X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    >>> r = geron_ridge_cost(X, [4.0, 7.0, 10.0], [4.0, 3.0], alpha=1.0)
    >>> round(r["mse"], 12)
    0.0
    >>> round(r["penalty"], 10)
    4.5
    >>> round(r["cost"], 10)
    4.5

    alpha = 0 leaves the plain MSE:

    >>> geron_ridge_cost(X, [4.0, 7.0, 10.0], [0.0, 0.0], alpha=0.0)["cost"]
    55.0
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    th = np.asarray(theta, dtype=float).ravel()
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty (m, n) matrix, got shape {A.shape}.")
    if A.shape[0] != yv.size:
        raise ValueError(f"X has {A.shape[0]} rows but y has {yv.size} entries.")
    if A.shape[1] != th.size:
        raise ValueError(f"X has {A.shape[1]} features but theta has {th.size} entries.")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(yv)) and np.all(np.isfinite(th))):
        raise ValueError("X, y and theta must be finite.")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be finite and non-negative, got {alpha}.")

    res = A @ th - yv
    mse = float(np.mean(res**2))
    w = th.copy()
    if intercept:
        w[0] = 0.0
    pen = float(0.5 * alpha * (w @ w))
    grad = (2.0 / A.shape[0]) * (A.T @ res) + alpha * w

    return RichResult(
        title="Ridge cost",
        summary_lines=[("Cost", mse + pen), ("MSE", mse), ("Penalty", pen)],
        payload={
            "cost": mse + pen,
            "mse": mse,
            "penalty": pen,
            "gradient": grad.tolist(),
            "estimate": mse + pen,
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grridg: J = MSE + (alpha/2) sum_{i>=1} theta_i^2; bias unpenalised, gradient included"
