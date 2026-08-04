# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Nesterov accelerated gradient (NAG)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_nesterov"]


def geron_nesterov(grads, v=None, beta=0.9, eta=0.001, theta=None):
    """
    Nesterov accelerated gradient (NAG).

    Formula: v <- beta*v + grad(theta - eta*beta*v); theta <- theta - eta*v

    The gradient is measured at the LOOK-AHEAD point, slightly ahead of
    the current parameters in the direction momentum is already carrying
    them. When momentum is pointing at the valley wall the look-ahead
    gradient already points back, so NAG damps the oscillation that plain
    momentum keeps building.

    Pass ``grads`` as a callable to have the look-ahead evaluated here
    (``theta`` is then required); pass an array only if you evaluated it
    at ``theta - eta*beta*v`` yourself.

    Parameters
    ----------
    grads : callable or array-like
        ``grads(theta_lookahead) -> gradient``, or the already-evaluated
        look-ahead gradient.
    v : array-like, optional
        Velocity from the previous step; default zeros.
    beta : float, default 0.9
        Momentum in [0, 1).
    eta : float, default 0.001
        Learning rate (positive).
    theta : array-like, optional
        Current parameters; required when ``grads`` is callable.

    Returns
    -------
    result : RichResult
        Keys: theta, v, lookahead, gradient, step, estimate, n, method.

    Examples
    --------
    On f(x) = x^2/2 (gradient x) from x = 1 with zero velocity the
    look-ahead is 1 itself, so v = 1 and x becomes 0.9:

    >>> r = geron_nesterov(lambda x: x, v=[0.0], beta=0.9, eta=0.1, theta=[1.0])
    >>> float(r["v"][0]), float(r["theta"][0])
    (1.0, 0.9)

    The next step looks ahead to 0.9 - 0.1*0.9*1 = 0.81:

    >>> r2 = geron_nesterov(lambda x: x, v=r["v"], beta=0.9, eta=0.1, theta=r["theta"])
    >>> float(r2["lookahead"][0]), round(float(r2["v"][0]), 12)
    (0.81, 1.71)
    >>> round(float(r2["theta"][0]), 12)
    0.729

    References
    ----------
    Geron Ch 11
    """
    b, lr = float(beta), float(eta)
    if not (0.0 <= b < 1.0):
        raise ValueError(f"geron_nesterov: beta must lie in [0, 1), got {beta!r}")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("geron_nesterov: eta must be a positive finite learning rate")

    if callable(grads):
        if theta is None:
            raise ValueError("geron_nesterov: theta is required when grads is a callable, to form the look-ahead")
        th = np.atleast_1d(np.asarray(theta, dtype=float))
        vv = np.zeros_like(th) if v is None else np.atleast_1d(np.asarray(v, dtype=float))
        if vv.shape != th.shape:
            raise ValueError(f"geron_nesterov: v has shape {vv.shape} but theta has shape {th.shape}")
        look = th - lr * b * vv
        g = np.atleast_1d(np.asarray(grads(look), dtype=float))
        if g.shape != th.shape:
            raise ValueError(f"geron_nesterov: grads returned shape {g.shape} for parameters of shape {th.shape}")
    else:
        g = np.atleast_1d(np.asarray(grads, dtype=float))
        th = np.zeros_like(g) if theta is None else np.atleast_1d(np.asarray(theta, dtype=float))
        vv = np.zeros_like(g) if v is None else np.atleast_1d(np.asarray(v, dtype=float))
        if vv.shape != g.shape or th.shape != g.shape:
            raise ValueError("geron_nesterov: v and theta must have the same shape as grads")
        look = th - lr * b * vv
    if g.size == 0:
        raise ValueError("geron_nesterov: gradient is empty")
    if not np.all(np.isfinite(g)):
        raise ValueError("geron_nesterov: gradient contains non-finite values")

    v_new = b * vv + g
    step = -lr * v_new
    theta_next = th + step
    return RichResult(
        title="Nesterov accelerated gradient",
        summary_lines=[("beta", b), ("Step L2 norm", float(np.linalg.norm(step)))],
        interpretation="The gradient is read ahead of the parameters, which damps momentum overshoot.",
        payload={
            "theta": theta_next,
            "theta_next": theta_next,
            "v": v_new,
            "lookahead": look,
            "gradient": g,
            "step": step,
            "estimate": theta_next,
            "n": int(g.size),
            "method": "Nesterov accelerated gradient with look-ahead evaluation",
        },
    )


def cheatsheet():
    return "hmnag: Nesterov accelerated gradient (NAG)"


# compact alias per ledger/NAMING.md
geronnesterov = geron_nesterov
