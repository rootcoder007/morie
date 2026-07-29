# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AdaMax: Adam variant using L-infinity norm."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_adamax"]


def geron_adamax(grads, m=None, u=None, b1=0.9, b2=0.999, eta=0.002, t=1, theta=None):
    """
    AdaMax: Adam variant using the L-infinity norm.

    Formula: u <- max(b2*u, |g|); theta <- theta - eta*m_hat/u

    Because `u` is an exponentially weighted max rather than a mean of
    squares, no bias correction is needed on the second moment -- only
    `m` is bias-corrected.

    Parameters
    ----------
    grads : array-like
        Gradient of the loss w.r.t. the parameters.
    m : array-like, optional
        First-moment accumulator; default zeros.
    u : array-like, optional
        Exponentially weighted infinity-norm accumulator; default zeros.
        Must be non-negative.
    b1, b2 : float
        Decay rates in [0, 1).
    eta : float
        Learning rate (positive).
    t : int
        1-based timestep for bias correction of m.
    theta : array-like, optional
        Current parameters; default zeros.

    Returns
    -------
    result : RichResult
        Keys: theta, step, m, u, m_hat, estimate, n, method.

    Examples
    --------
    >>> r = geron_adamax([1.0], m=[0.0], u=[0.0], eta=0.1, t=1)
    >>> round(float(r["m"][0]), 12), float(r["u"][0])
    (0.1, 1.0)
    >>> round(float(r["m_hat"][0]), 9)
    1.0
    >>> round(float(r["step"][0]), 9)
    -0.1
    >>> r2 = geron_adamax([0.5], m=[0.1], u=[1.0], b1=0.9, b2=0.999, eta=0.1, t=2)
    >>> round(float(r2["u"][0]), 6)
    0.999
    >>> round(float(r2["step"][0]), 6)
    -0.073758

    References
    ----------
    Géron Ch 11
    """
    g = np.atleast_1d(np.asarray(grads, dtype=float))
    if g.size == 0:
        raise ValueError("geron_adamax: grads is empty")
    if not np.all(np.isfinite(g)):
        raise ValueError("geron_adamax: grads contains non-finite values")
    mm = np.zeros_like(g) if m is None else np.atleast_1d(np.asarray(m, dtype=float))
    uu = np.zeros_like(g) if u is None else np.atleast_1d(np.asarray(u, dtype=float))
    th = np.zeros_like(g) if theta is None else np.atleast_1d(np.asarray(theta, dtype=float))
    for name, arr in (("m", mm), ("u", uu), ("theta", th)):
        if arr.shape != g.shape:
            raise ValueError(f"geron_adamax: {name} has shape {arr.shape} but grads has shape {g.shape}")
    if np.any(uu < 0):
        raise ValueError("geron_adamax: u must be non-negative (it tracks |g|)")
    beta1, beta2, lr = float(b1), float(b2), float(eta)
    if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
        raise ValueError(f"geron_adamax: b1 and b2 must lie in [0, 1), got b1={beta1}, b2={beta2}")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("geron_adamax: eta must be a positive finite learning rate")
    step_t = int(t)
    if step_t < 1:
        raise ValueError("geron_adamax: t must be a 1-based timestep >= 1")

    m_new = beta1 * mm + (1.0 - beta1) * g
    u_new = np.maximum(beta2 * uu, np.abs(g))
    m_hat = m_new / (1.0 - beta1**step_t)
    if np.any(u_new == 0):
        # u = 0 means the coordinate has never seen a non-zero gradient;
        # its step is exactly 0 rather than 0/0.
        step = np.where(u_new == 0, 0.0, -lr * m_hat / np.where(u_new == 0, 1.0, u_new))
    else:
        step = -lr * m_hat / u_new
    theta_next = th + step

    return RichResult(
        title="AdaMax step",
        summary_lines=[("Timestep", step_t), ("Step L2 norm", float(np.linalg.norm(step)))],
        payload={
            "theta": theta_next,
            "theta_next": theta_next,
            "step": step,
            "m": m_new,
            "u": u_new,
            "m_hat": m_hat,
            "t": step_t,
            "estimate": float(np.linalg.norm(step)),
            "n": int(g.size),
            "method": "AdaMax (Adam with an L-infinity second moment)",
        },
    )


def cheatsheet():
    return "hmadmx: AdaMax: Adam variant using L-infinity norm"
