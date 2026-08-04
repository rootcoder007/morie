# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Momentum optimization: accumulates exponentially-decaying past gradients."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_momentum"]

_METHOD = "Momentum optimization step"


def geron_momentum(grads, v=None, beta=0.9, eta=0.01, theta=None, nesterov=False):
    """
    Momentum optimization: accumulates exponentially-decaying past gradients.

    Formula: v <- beta*v + grad; theta <- theta - eta*v

    One momentum step, in the spec's convention where the new gradient
    enters with weight 1 rather than ``1 - beta``.  On a constant
    gradient ``g`` the velocity converges to ``g/(1 - beta)``, so
    ``beta = 0.9`` reaches a terminal speed ten times plain gradient
    descent -- that terminal step size is returned, because it is the
    quantity that decides whether a learning rate that was fine without
    momentum will now diverge.

    With ``nesterov=True`` the gradient is assumed to have been measured
    at the look-ahead point ``theta - eta*beta*v`` and the update uses
    ``theta <- theta - eta*(beta*v_new + grad)``.

    Parameters
    ----------
    grads : array-like
        Gradient of the loss w.r.t. the parameters.
    v : array-like, optional
        Velocity from the previous step; default zeros.
    beta : float
        Momentum coefficient in [0, 1).
    eta : float
        Learning rate (positive).
    theta : array-like, optional
        Current parameters; default zeros.
    nesterov : bool
        Use the Nesterov accelerated form.

    Returns
    -------
    result : RichResult
        Keys: theta, theta_next, step, v, terminal_step, estimate, n, method.

    Examples
    --------
    First step from rest: ``v = 0.9*0 + 1 = 1``, ``step = -0.1``.

    >>> r = geron_momentum([1.0], v=[0.0], beta=0.9, eta=0.1)
    >>> float(r["v"][0]), float(r["step"][0])
    (1.0, -0.1)

    Second step with the same gradient: ``v = 0.9*1 + 1 = 1.9``.

    >>> r2 = geron_momentum([1.0], v=r["v"], beta=0.9, eta=0.1)
    >>> round(float(r2["v"][0]), 12), round(float(r2["step"][0]), 12)
    (1.9, -0.19)

    Terminal velocity on a constant gradient is ``g/(1-beta) = 10``, so
    the terminal step is ``0.1 * 10 = 1``:

    >>> round(float(r["terminal_step"][0]), 12)
    1.0

    ``beta = 0`` is plain gradient descent:

    >>> float(geron_momentum([2.0], beta=0.0, eta=0.5)["step"][0])
    -1.0

    References
    ----------
    Géron Ch 11
    """
    g = np.atleast_1d(np.asarray(grads, dtype=float))
    if g.size == 0:
        raise ValueError("geron_momentum: grads is empty")
    if not np.all(np.isfinite(g)):
        raise ValueError("geron_momentum: grads contains non-finite values")
    vv = np.zeros_like(g) if v is None else np.atleast_1d(np.asarray(v, dtype=float))
    th = np.zeros_like(g) if theta is None else np.atleast_1d(np.asarray(theta, dtype=float))
    for name, arr in (("v", vv), ("theta", th)):
        if arr.shape != g.shape:
            raise ValueError(f"geron_momentum: {name} has shape {arr.shape} but grads has shape {g.shape}")
    b, lr = float(beta), float(eta)
    if not (0.0 <= b < 1.0):
        raise ValueError(f"geron_momentum: beta must lie in [0, 1) or the velocity never decays, got {beta!r}")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"geron_momentum: eta must be a positive finite learning rate, got {eta!r}")

    v_new = b * vv + g
    if nesterov:
        step = -lr * (b * v_new + g)
    else:
        step = -lr * v_new
    theta_next = th + step
    terminal = lr * g / (1.0 - b)

    return RichResult(
        title="Momentum step",
        summary_lines=[
            ("beta", b),
            ("Step L2 norm", float(np.linalg.norm(step))),
            ("Terminal step norm", float(np.linalg.norm(terminal))),
        ],
        interpretation=(
            f"On a constant gradient the step size approaches 1/(1-beta) = {1.0 / (1.0 - b):.4g} times "
            "the plain gradient-descent step, so momentum effectively multiplies the learning rate."
        ),
        payload={
            "theta": theta_next,
            "theta_next": theta_next,
            "step": step,
            "v": v_new,
            "terminal_step": terminal,
            "beta": b,
            "estimate": float(np.linalg.norm(step)),
            "n": int(g.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmom: momentum v <- beta*v + g; theta <- theta - eta*v (terminal speed g/(1-beta))"


# compact alias per ledger/NAMING.md
geronmomentum = geron_momentum
