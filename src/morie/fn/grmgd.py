# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mini-batch gradient descent for linear regression."""

import numpy as np

from ._richresult import RichResult
from .grdlm import geron_dataloader_minibatch
from .grmse import geron_linreg_mse_cost
from .grn007 import geron_ch4_mse_gradient_vector

__all__ = ["geron_minibatch_gradient_descent"]

_METHOD = "Mini-batch gradient descent (linear regression)"


def geron_minibatch_gradient_descent(X, y, theta, eta, b, n_iter, seed=0):
    r"""Run ``n_iter`` mini-batch steps.

    .. math::
        \theta_{t+1} = \theta_t - \eta\,\frac{2}{b}
        X_b^{\mathsf T}(X_b \theta_t - y_b)

    The ``2/b`` -- averaging over the *batch*, not the dataset -- is
    what keeps a mini-batch step the same size as a full-batch step.
    Sum instead of average and the effective learning rate scales with
    the batch size.

    Batches come from
    :func:`morie.fn.grdlm.geron_dataloader_minibatch` (reshuffled every
    epoch) and each gradient from
    :func:`morie.fn.grn007.geron_ch4_mse_gradient_vector`, so this
    function is only the loop.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    theta : array-like, shape (n,)
        Starting parameters.
    eta : float
        Positive learning rate.
    b : int
        Batch size, ``1 <= b <= m``.
    n_iter : int
        Number of *steps* (not epochs), at least 1.
    seed : int, optional

    Returns
    -------
    RichResult
        Payload keys ``theta``, ``cost_history`` (full-data MSE after
        each step, length ``n_iter + 1``), ``theta_history``,
        ``final_cost``, ``initial_cost``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 4, Mini-batch Gradient Descent section.

    Examples
    --------
    Fitting ``y = x`` from a zero start: the cost falls and the slope
    heads for 1.

    >>> X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
    >>> y = [1.0, 2.0, 3.0, 4.0]
    >>> r = geron_minibatch_gradient_descent(X, y, [0.0, 0.0], eta=0.05, b=2,
    ...                                      n_iter=200, seed=0)
    >>> r["final_cost"] < 1e-4
    True
    >>> round(r["theta"][1], 2)
    1.0

    One step with the full dataset as the batch is exactly a batch
    gradient step: from ``theta = 0`` the gradient is ``[-5, -15]``, so
    ``eta = 0.01`` moves to ``[0.05, 0.15]``:

    >>> r2 = geron_minibatch_gradient_descent(X, y, [0.0, 0.0], eta=0.01, b=4,
    ...                                       n_iter=1, seed=0)
    >>> [round(v, 10) for v in r2["theta"]]
    [0.05, 0.15]
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    y_arr = np.asarray(y, dtype=float).ravel()
    th = np.asarray(theta, dtype=float).ravel().copy()
    start = geron_linreg_mse_cost(A, y_arr, th)      # validates shapes
    m = A.shape[0]
    b = int(b)
    if not (1 <= b <= m):
        raise ValueError(f"b must lie in [1, {m}], got {b}.")
    n_iter = int(n_iter)
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}.")
    eta = float(eta)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"eta must be a positive finite float, got {eta}.")

    costs = [start["cost"]]
    hist = [th.tolist()]
    epoch = 0
    queue = []
    for _ in range(n_iter):
        if not queue:
            queue = list(geron_dataloader_minibatch(m, b, shuffle=True,
                                                    seed=int(seed) + epoch)["batches"])
            epoch += 1
        idx = queue.pop(0)
        g = geron_ch4_mse_gradient_vector(A[idx], y_arr[idx], th)["gradient"]
        th = th - eta * np.asarray(g, dtype=float)
        if not np.all(np.isfinite(th)):
            raise ValueError(
                f"parameters diverged to non-finite values after {len(costs)} steps; "
                f"eta = {eta} is too large for this data."
            )
        costs.append(geron_linreg_mse_cost(A, y_arr, th)["cost"])
        hist.append(th.tolist())

    return RichResult(
        title="Mini-batch gradient descent",
        summary_lines=[("Steps", n_iter), ("Batch size", b),
                       ("Cost", costs[-1])],
        payload={
            "theta": th.tolist(),
            "cost_history": costs,
            "theta_history": hist,
            "initial_cost": costs[0],
            "final_cost": costs[-1],
            "eta": eta,
            "batch_size": b,
            "estimate": th.tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmgd: theta -= eta*(2/b) X_b^T (X_b theta - y_b); batches via grdlm, gradient via grn007"
