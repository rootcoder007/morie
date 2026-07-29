# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch gradient descent on linear-regression MSE."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_batch_gradient_descent"]

_METHOD = "Batch gradient descent (linear regression MSE)"


def geron_batch_gradient_descent(X, y, theta, eta, n_iter):
    r"""Run ``n_iter`` full-batch gradient-descent steps.

    .. math::
        \theta \leftarrow \theta - \eta \frac{2}{m} X^{\top}(X\theta - y)

    Every step touches the whole training set, so the loss curve is
    smooth and monotone *provided* :math:`\eta < 2/\lambda_{\max}` where
    :math:`\lambda_{\max}` is the largest eigenvalue of
    :math:`\frac{2}{m}X^{\top}X`.  That bound is reported so a diverging
    run can be diagnosed rather than guessed at.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix -- add the bias column yourself if you want an
        intercept.
    y : array-like, shape (m,)
        Targets.
    theta : array-like, shape (n,)
        Starting parameters.
    eta : float
        Learning rate, positive.
    n_iter : int
        Number of steps, at least 1.

    Returns
    -------
    RichResult
        Payload keys ``theta`` (final), ``theta_path``,
        ``loss_history`` (MSE after each step, plus the initial value),
        ``gradient`` (at the final point), ``eta_max_stable``,
        ``converged``, ``estimate`` (final MSE), ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-6 (Batch Gradient Descent step).

    Examples
    --------
    One step on ``y = 2x`` from ``theta = 0``:

    >>> r = geron_batch_gradient_descent([[1.0], [2.0]], [2.0, 4.0], [0.0],
    ...                                  eta=0.1, n_iter=1)
    >>> r["theta"]
    [1.0]
    >>> round(r["loss_history"][0], 6)
    10.0
    >>> round(r["loss_history"][1], 6)
    2.5

    Running it out recovers the least-squares solution:

    >>> r2 = geron_batch_gradient_descent([[1.0], [2.0]], [2.0, 4.0], [0.0],
    ...                                   eta=0.1, n_iter=200)
    >>> round(r2["theta"][0], 8)
    2.0
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    if X.shape[0] != y.size:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.size} entries."
        )
    if X.shape[1] != theta.size:
        raise ValueError(
            f"X has {X.shape[1]} columns but theta has {theta.size} entries."
        )
    if X.size == 0:
        raise ValueError("X is empty.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(theta)):
        raise ValueError("X, y and theta must all be finite.")
    eta = float(eta)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"eta must be a positive finite float, got {eta}.")
    n_iter = int(n_iter)
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}.")

    m = X.shape[0]
    H = (2.0 / m) * (X.T @ X)
    lam_max = float(np.max(np.linalg.eigvalsh(H))) if X.shape[1] else 0.0
    eta_max = float("inf") if lam_max <= 0 else 2.0 / lam_max

    def mse(th):
        r = X @ th - y
        return float(r @ r / m)

    path = [theta.tolist()]
    losses = [mse(theta)]
    grad = np.zeros_like(theta)
    for _ in range(n_iter):
        grad = (2.0 / m) * (X.T @ (X @ theta - y))
        theta = theta - eta * grad
        if not np.all(np.isfinite(theta)):
            raise ValueError(
                f"gradient descent diverged; eta={eta} exceeds the stability "
                f"bound {eta_max:.6g}."
            )
        path.append(theta.tolist())
        losses.append(mse(theta))

    return RichResult(
        title="Batch gradient descent",
        summary_lines=[("Final MSE", losses[-1]), ("Steps", n_iter)],
        payload={
            "theta": theta.tolist(),
            "theta_path": path,
            "loss_history": losses,
            "gradient": grad.tolist(),
            "eta": eta,
            "eta_max_stable": eta_max,
            "converged": bool(losses[-1] <= losses[0]),
            "estimate": losses[-1],
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grbgd: batch GD -- theta -= eta*(2/m)*X^T(X theta - y), full loss curve returned"
