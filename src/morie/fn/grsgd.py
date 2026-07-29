# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stochastic gradient descent on linear-regression MSE, one sample per step."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_stochastic_gradient_descent"]

_METHOD = "Stochastic gradient descent (single-sample MSE gradient)"


def geron_stochastic_gradient_descent(X, y, theta, eta, n_iter, seed=42, t0=None, t1=None):
    r"""Run ``n_iter`` single-sample gradient steps.

    The per-sample MSE gradient of :math:`(\mathbf{x}^{(i)\top}\theta -
    y^{(i)})^2` is

    .. math::
        \nabla = 2\,\mathbf{x}^{(i)}
                 \bigl(\mathbf{x}^{(i)\top}\theta - y^{(i)}\bigr),
        \qquad
        \theta \leftarrow \theta - \eta_t \nabla

    (the worklist line prints ``y^(i)`` where the book has
    :math:`\mathbf{x}^{(i)}` in the two outer factors -- the gradient
    above is Géron's).  Sample order comes from the reproducible LCG
    ``s = (1664525 s + 1013904223) mod 2^32``, so two runs with the same
    seed agree exactly.  Supplying ``t0``/``t1`` switches on the simulated
    annealing schedule :math:`\eta_t = t_0/(t + t_1)`, without which SGD
    bounces around the optimum forever instead of settling into it.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    theta : array-like, shape (n,)
        Starting parameters.
    eta : float
        Learning rate; ignored when ``t0`` and ``t1`` are both given.
    n_iter : int
        Number of single-sample steps.
    seed : int, optional
        LCG seed for the sample order.
    t0, t1 : float, optional
        Learning schedule numerator and offset.

    Returns
    -------
    RichResult
        Payload keys ``theta``, ``path`` (theta after every step),
        ``cost_path`` (full-sample MSE), ``learning_rates``,
        ``sample_order``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Stochastic Gradient Descent section.

    Examples
    --------
    One step by hand: ``x = [1, 0]``, residual ``0*1 - 4 = -4``,
    gradient ``2 * [1, 0] * (-4) = [-8, 0]``, so with ``eta = 0.1``
    theta moves to ``[0.8, 0]``:

    >>> X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    >>> r = geron_stochastic_gradient_descent(X, [4.0, 7.0, 10.0], [0.0, 0.0],
    ...                                       eta=0.1, n_iter=1, seed=1)
    >>> r["sample_order"]
    [0]
    >>> [round(v, 10) for v in r["theta"]]
    [0.8, 0.0]

    Enough steps and the cost falls:

    >>> long = geron_stochastic_gradient_descent(X, [4.0, 7.0, 10.0], [0.0, 0.0],
    ...                                          eta=0.05, n_iter=200, seed=7)
    >>> long["cost_path"][-1] < long["cost_path"][0]
    True
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    th = np.asarray(theta, dtype=float).ravel().copy()
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty (m, n) matrix, got shape {A.shape}.")
    if A.shape[0] != yv.size:
        raise ValueError(f"X has {A.shape[0]} rows but y has {yv.size} entries.")
    if A.shape[1] != th.size:
        raise ValueError(f"X has {A.shape[1]} features but theta has {th.size} entries.")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(yv)) and np.all(np.isfinite(th))):
        raise ValueError("X, y and theta must be finite.")
    n_iter = int(n_iter)
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}.")
    scheduled = t0 is not None and t1 is not None
    if (t0 is None) != (t1 is None):
        raise ValueError("t0 and t1 must be supplied together or not at all.")
    if scheduled:
        t0, t1 = float(t0), float(t1)
        if t0 <= 0 or t1 <= 0:
            raise ValueError(f"t0 and t1 must be positive, got {t0} and {t1}.")
    else:
        eta = float(eta)
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError(f"eta must be a positive finite float, got {eta}.")

    m = A.shape[0]
    s = int(seed) % 2**32
    path, costs, rates, order = [], [], [], []
    costs.append(float(np.mean((A @ th - yv) ** 2)))
    for t in range(n_iter):
        s = (1664525 * s + 1013904223) % 2**32
        u = (s + 0.5) / 2**32
        i = int(u * m)
        if i == m:
            i = m - 1
        lr = t0 / (t + t1) if scheduled else eta
        grad = 2.0 * A[i] * (A[i] @ th - yv[i])
        th = th - lr * grad
        if not np.all(np.isfinite(th)):
            raise ValueError(
                f"theta diverged at step {t} (eta={lr:g} too large for this data)."
            )
        path.append(th.tolist())
        costs.append(float(np.mean((A @ th - yv) ** 2)))
        rates.append(float(lr))
        order.append(i)

    return RichResult(
        title="Stochastic gradient descent",
        summary_lines=[("Steps", n_iter), ("Final MSE", costs[-1])],
        payload={
            "theta": th.tolist(),
            "path": path,
            "cost_path": costs,
            "learning_rates": rates,
            "sample_order": order,
            "estimate": th.tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grsgd: theta -= eta * 2 x_i (x_i^T theta - y_i), one LCG-drawn sample per step; t0/t1 anneals eta"
