# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Elastic net cost combining L1 and L2 with ratio r."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_elastic_net"]


def geron_elastic_net(X, y, theta, alpha, r, fit_intercept=True):
    """
    Elastic net cost combining L1 and L2 with ratio r.

    Formula: J = MSE + r*alpha*sum|theta_i| + ((1-r)/2)*alpha*sum theta_i^2

    ``r = 1`` reduces to lasso and ``r = 0`` to ridge, so both endpoints
    are checked against the dedicated implementations. With
    ``fit_intercept=True`` the first entry of ``theta`` is the bias and is
    left out of both penalties, matching the convention Géron uses.

    A subgradient is returned as well; at a coordinate where
    ``theta_i == 0`` the L1 part is reported as 0, the minimum-norm
    element of the subdifferential.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix; if ``fit_intercept`` it must NOT already contain a
        bias column -- one is prepended.
    y : array-like, shape (m,)
        Targets.
    theta : array-like
        Parameters; length ``n + 1`` with an intercept, else ``n``.
    alpha : float
        Overall penalty weight, non-negative.
    r : float
        L1 ratio in [0, 1].
    fit_intercept : bool, default True
        Treat ``theta[0]`` as an unpenalised bias.

    Returns
    -------
    result : RichResult
        Keys: cost, mse, l1_penalty, l2_penalty, gradient, estimate,
        n, method.

    Examples
    --------
    A perfect fit leaves only the penalties: theta = (0, 2) on y = 2x.

    >>> r0 = geron_elastic_net([[1.0], [2.0]], [2.0, 4.0], [0.0, 2.0], alpha=1.0, r=0.5)
    >>> round(r0["mse"], 12), round(r0["l1_penalty"], 12), round(r0["l2_penalty"], 12)
    (0.0, 1.0, 1.0)
    >>> round(r0["cost"], 12)
    2.0

    r = 1 is lasso, r = 0 is ridge:

    >>> round(geron_elastic_net([[1.0]], [1.0], [0.0, 3.0], 2.0, 1.0)["cost"], 12)
    10.0
    >>> round(geron_elastic_net([[1.0]], [1.0], [0.0, 3.0], 2.0, 0.0)["cost"], 12)
    13.0

    References
    ----------
    Géron Ch 4
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    th = np.atleast_1d(np.asarray(theta, dtype=float))
    a = float(alpha)
    ratio = float(r)
    if X.size == 0 or y.size == 0:
        raise ValueError("geron_elastic_net: X and y must be non-empty")
    if X.shape[0] != y.size:
        raise ValueError(f"geron_elastic_net: X has {X.shape[0]} rows but y has {y.size} entries")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(th)):
        raise ValueError("geron_elastic_net: X, y and theta must be finite")
    if not np.isfinite(a) or a < 0:
        raise ValueError(f"geron_elastic_net: alpha must be non-negative and finite, got {alpha!r}")
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"geron_elastic_net: r must lie in [0, 1], got {r!r}")

    Xd = np.hstack([np.ones((X.shape[0], 1)), X]) if fit_intercept else X
    if th.size != Xd.shape[1]:
        raise ValueError(
            f"geron_elastic_net: theta has {th.size} entries but the design matrix has "
            f"{Xd.shape[1]} columns (fit_intercept={fit_intercept})"
        )

    m = Xd.shape[0]
    resid = Xd @ th - y
    mse = float(np.mean(resid**2))
    pen = th.copy()
    if fit_intercept:
        pen[0] = 0.0
    l1 = float(ratio * a * np.sum(np.abs(pen)))
    l2 = float(0.5 * (1.0 - ratio) * a * np.sum(pen**2))
    cost = mse + l1 + l2

    grad = (2.0 / m) * (Xd.T @ resid) + ratio * a * np.sign(pen) + (1.0 - ratio) * a * pen

    return RichResult(
        title="Elastic net cost",
        summary_lines=[("Cost", cost), ("MSE", mse), ("L1", l1), ("L2", l2)],
        interpretation="r = 1 is lasso, r = 0 is ridge; the intercept is never penalised.",
        payload={
            "cost": cost,
            "mse": mse,
            "l1_penalty": l1,
            "l2_penalty": l2,
            "penalty": l1 + l2,
            "gradient": grad,
            "alpha": a,
            "r": ratio,
            "estimate": cost,
            "n": int(m),
            "method": "elastic net J = MSE + r*alpha*L1 + (1-r)/2*alpha*L2",
        },
    )


def cheatsheet():
    return "hmenet: Elastic net cost combining L1 and L2 with ratio r"
