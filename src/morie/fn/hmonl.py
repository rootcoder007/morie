# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Online learning: sequential updates from a data stream."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_online_learning"]


def geron_online_learning(X_stream, y_stream, eta=0.1, theta=None, decay=0.0):
    """
    Online learning: sequentially update a model with streaming data.

    Formula: theta_{t+1} = theta_t - eta_t * grad L(theta_t; x_t, y_t)

    Each instance is seen once, in order, and then discarded, so memory
    is O(parameters) rather than O(data) -- the property that lets the
    model outlive the dataset. The price is that the model also FORGETS
    in order: if the stream drifts, the last instances dominate, which is
    the failure Geron calls out. The loss recorded at each step is the
    loss BEFORE that step's update, so the sum is the honest prequential
    (predict-then-update) error, not a training-set fit.

    A ``decay`` d gives the Robbins-Monro schedule eta_t = eta/(1 + d*t):
    a constant rate never converges on a stationary stream, and a
    decaying one cannot track a drifting one.

    Parameters
    ----------
    X_stream : array-like, shape (T, n)
    y_stream : array-like, shape (T,)
    eta : float, default 0.1
        Base learning rate (positive).
    theta : array-like, optional
        Starting parameters; default zeros.
    decay : float, default 0.0
        Learning-rate decay (non-negative).

    Returns
    -------
    result : RichResult
        Keys: theta, trajectory, losses, cumulative_loss, mean_loss,
        estimate, n, method.

    Examples
    --------
    Squared loss with gradient 2(x.theta - y)x. From theta = 0 on
    x = 1, y = 1 with eta = 0.5 the step is -0.5 * (-2) = 1:

    >>> r = geron_online_learning([[1.0], [1.0]], [1.0, 1.0], eta=0.5)
    >>> [float(v) for v in r["losses"]]
    [1.0, 0.0]
    >>> float(r["theta"][0])
    1.0

    The second instance is already predicted exactly, so the parameters
    stop moving:

    >>> [float(v) for v in r["trajectory"][:, 0]]
    [0.0, 1.0, 1.0]

    References
    ----------
    Geron Ch 1
    """
    A = np.asarray(X_stream, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_online_learning: X_stream must be a non-empty 2-D array, got shape {A.shape}")
    yv = np.atleast_1d(np.asarray(y_stream, dtype=float)).ravel()
    if yv.size != A.shape[0]:
        raise ValueError(f"geron_online_learning: X_stream has {A.shape[0]} rows but y_stream has {yv.size}")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yv)):
        raise ValueError("geron_online_learning: the stream contains non-finite values")
    base = float(eta)
    if not np.isfinite(base) or base <= 0:
        raise ValueError(f"geron_online_learning: eta must be positive and finite, got {eta!r}")
    d = float(decay)
    if d < 0:
        raise ValueError(f"geron_online_learning: decay must be non-negative, got {decay!r}")
    th = np.zeros(A.shape[1]) if theta is None else np.atleast_1d(np.asarray(theta, dtype=float)).astype(float).copy()
    if th.size != A.shape[1]:
        raise ValueError(f"geron_online_learning: theta has {th.size} entries but X_stream has {A.shape[1]} columns")

    T = A.shape[0]
    traj = np.empty((T + 1, th.size))
    traj[0] = th
    losses = np.empty(T)
    for t in range(T):
        pred = float(A[t] @ th)
        err = pred - yv[t]
        losses[t] = err * err
        rate = base / (1.0 + d * t)
        th = th - rate * 2.0 * err * A[t]
        traj[t + 1] = th

    return RichResult(
        title="Online (streaming) learning",
        summary_lines=[("Instances", int(T)), ("Prequential loss", float(losses.sum())), ("Final rate", base / (1.0 + d * max(T - 1, 0)))],
        interpretation="Losses are pre-update, so their sum is an honest predict-then-update error.",
        payload={
            "theta": th,
            "trajectory": traj,
            "losses": losses,
            "cumulative_loss": float(losses.sum()),
            "mean_loss": float(losses.mean()),
            "estimate": th,
            "n": int(T),
            "method": "Online SGD on squared loss with an optional Robbins-Monro decay",
        },
    )


def cheatsheet():
    return "hmonl: Online learning by sequential SGD over a stream"
