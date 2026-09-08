# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Nesterov accelerated gradient step."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_nesterov_accelerated_gradient"]

_METHOD = "Nesterov accelerated gradient step"


def geron_nesterov_accelerated_gradient(theta, grad_fn, v, eta, beta=0.9, n_steps=1):
    r"""Momentum that measures the gradient where it is about to be.

    .. math::
        g &= \nabla J(\theta_t - \eta\beta\mathbf{v}_t)\\
        \mathbf{v}_{t+1} &= \beta\mathbf{v}_t + g\\
        \theta_{t+1} &= \theta_t - \eta\,\mathbf{v}_{t+1}

    Plain momentum evaluates the gradient at :math:`\theta_t`; Nesterov
    evaluates it at the look-ahead point the momentum is already carrying
    you to.  When the step overshoots, the look-ahead gradient points
    back before you arrive, which damps the oscillation instead of
    amplifying it -- that is the whole of the "acceleration".

    ``grad_fn`` is caller-supplied and its contract is enforced: it must
    return a finite array with the same shape as ``theta``.  A silently
    mis-shaped gradient would broadcast and produce a wrong step of the
    right dtype.

    Parameters
    ----------
    theta : array-like, shape (p,)
    grad_fn : callable
        ``grad_fn(theta) -> array of shape (p,)``.
    v : array-like, shape (p,)
        Momentum accumulator.
    eta : float
        Positive learning rate.
    beta : float, optional
        Momentum in ``[0, 1)``.
    n_steps : int, optional
        Number of steps to run.

    Returns
    -------
    RichResult
        Payload keys ``theta_new``, ``v_new``, ``lookahead``,
        ``gradient``, ``path``, ``step``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 11, Nesterov Accelerated Gradient section.

    Examples
    --------
    ``J = theta^2``, so ``grad = 2 theta``.  From ``theta = 1``,
    ``v = 0``: the look-ahead is 1 (no momentum yet), ``g = 2``,
    ``v = 2``, and ``eta = 0.1`` puts theta at ``1 - 0.2 = 0.8``.

    >>> r = geron_nesterov_accelerated_gradient([1.0], lambda t: 2 * t, [0.0], eta=0.1)
    >>> r["lookahead"], r["gradient"], r["v_new"]
    ([1.0], [2.0], [2.0])
    >>> [round(x, 10) for x in r["theta_new"]]
    [0.8]

    Second step: look-ahead is ``0.8 - 0.1*0.9*2 = 0.62``, so
    ``g = 1.24``, ``v = 1.8 + 1.24 = 3.04``, theta ``= 0.8 - 0.304``:

    >>> r2 = geron_nesterov_accelerated_gradient([0.8], lambda t: 2 * t, [2.0], eta=0.1)
    >>> [round(x, 10) for x in r2["theta_new"]]
    [0.496]
    """
    th = np.asarray(theta, dtype=float).ravel()
    vv = np.asarray(v, dtype=float).ravel()
    if th.size == 0:
        raise ValueError("theta is empty.")
    if vv.shape != th.shape:
        raise ValueError(f"v has {vv.size} entries but theta has {th.size}.")
    if not np.all(np.isfinite(th)) or not np.all(np.isfinite(vv)):
        raise ValueError("theta and v must be finite.")
    if not callable(grad_fn):
        raise ValueError(f"grad_fn must be callable, got {type(grad_fn).__name__}.")
    eta = float(eta)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"eta must be a positive finite float, got {eta}.")
    beta = float(beta)
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must lie in [0, 1), got {beta}.")
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}.")

    path, look, grads = [], None, None
    for _ in range(n_steps):
        ahead = th - eta * beta * vv
        g = np.asarray(grad_fn(ahead), dtype=float).ravel()
        if g.shape != th.shape:
            raise ValueError(
                f"grad_fn returned shape {g.shape} but theta has shape {th.shape}."
            )
        if not np.all(np.isfinite(g)):
            raise ValueError("grad_fn returned non-finite values.")
        vv = beta * vv + g
        th = th - eta * vv
        path.append(th.tolist())
        look, grads = ahead, g

    step = eta * vv
    return RichResult(
        title="Nesterov accelerated gradient",
        summary_lines=[("Steps", n_steps), ("Step L2 norm", float(np.linalg.norm(step)))],
        payload={
            "theta_new": th.tolist(),
            "v_new": vv.tolist(),
            "lookahead": look.tolist(),
            "gradient": grads.tolist(),
            "path": path,
            "step": step.tolist(),
            "estimate": th.tolist(),
            "n": int(th.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grnag: g at theta - eta*beta*v (look-ahead); v = beta v + g; theta -= eta v"
