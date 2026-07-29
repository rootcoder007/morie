# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gradient of linear-regression MSE cost."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_batch_gd_grad"]


def geron_batch_gd_grad(X, y, theta, eta=None):
    """
    Gradient of linear-regression MSE cost.

    Formula: grad J(theta) = (2/m) X^T (X theta - y)

    Parameters
    ----------
    X : array-like, shape (m, k)
        Design matrix (add your own intercept column if wanted).
    y : array-like, shape (m,)
        Targets.
    theta : array-like, shape (k,)
        Current parameter vector.
    eta : float, optional
        Learning rate. When given, one batch gradient-descent step
        ``theta - eta * grad`` is also returned.

    Returns
    -------
    result : RichResult
        Keys: gradient, cost, residuals, theta_next, estimate, n, method.

    Examples
    --------
    >>> r = geron_batch_gd_grad([[1.0, 1.0], [1.0, 2.0]], [1.0, 2.0], [0.0, 0.0])
    >>> [float(g) for g in r["gradient"]]
    [-3.0, -5.0]
    >>> float(r["cost"])
    2.5
    >>> r2 = geron_batch_gd_grad([[1.0, 1.0], [1.0, 2.0]], [1.0, 2.0], [0.0, 0.0], eta=0.1)
    >>> [round(float(t), 4) for t in r2["theta_next"]]
    [0.3, 0.5]

    References
    ----------
    Géron Ch 4
    """
    Xm = np.asarray(X, dtype=float)
    if Xm.ndim == 1:
        Xm = Xm.reshape(-1, 1)
    if Xm.ndim != 2:
        raise ValueError(f"geron_batch_gd_grad: X must be 2-D, got ndim={Xm.ndim}")
    yv = np.asarray(y, dtype=float).ravel()
    th = np.asarray(theta, dtype=float).ravel()
    m, k = Xm.shape
    if m == 0:
        raise ValueError("geron_batch_gd_grad: X has no rows")
    if yv.size != m:
        raise ValueError(f"geron_batch_gd_grad: X has {m} rows but y has {yv.size} entries")
    if th.size != k:
        raise ValueError(f"geron_batch_gd_grad: X has {k} columns but theta has {th.size} entries")
    if not (np.all(np.isfinite(Xm)) and np.all(np.isfinite(yv)) and np.all(np.isfinite(th))):
        raise ValueError("geron_batch_gd_grad: X, y and theta must all be finite")

    resid = Xm @ th - yv
    grad = (2.0 / m) * (Xm.T @ resid)
    cost = float(resid @ resid / m)

    theta_next = None
    if eta is not None:
        e = float(eta)
        if not np.isfinite(e) or e <= 0:
            raise ValueError("geron_batch_gd_grad: eta must be a positive finite learning rate")
        theta_next = th - e * grad

    return RichResult(
        title="Batch gradient descent -- MSE gradient",
        summary_lines=[("MSE cost", cost), ("Gradient norm", float(np.linalg.norm(grad)))],
        payload={
            "gradient": grad,
            "cost": cost,
            "residuals": resid,
            "theta": th,
            "theta_next": theta_next,
            "grad_norm": float(np.linalg.norm(grad)),
            "estimate": cost,
            "n": int(m),
            "method": "Gradient of linear-regression MSE cost, (2/m) X^T (X theta - y)",
        },
    )


def cheatsheet():
    return "hmbgdg: Gradient of linear-regression MSE cost"
