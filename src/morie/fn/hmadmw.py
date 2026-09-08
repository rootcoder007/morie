# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AdamW: decoupled weight decay."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_adamw"]


def geron_adamw(grads, m=None, v=None, b1=0.9, b2=0.999, eta=0.001, wd=0.01, eps=1e-8, t=1, theta=None):
    """
    AdamW: decoupled weight decay.

    Formula: theta <- theta - eta*m_hat/(sqrt(v_hat)+eps) - eta*wd*theta

    The decay term is applied straight to the parameters rather than being
    folded into the gradient, so it is *not* rescaled by the adaptive
    denominator -- that decoupling is the whole point of AdamW.

    Parameters
    ----------
    grads : array-like
        Gradient of the loss w.r.t. the parameters (no L2 term inside).
    m, v : array-like, optional
        Moment accumulators; default zeros.
    b1, b2 : float
        Exponential decay rates in [0, 1).
    eta : float
        Learning rate (positive).
    wd : float
        Weight-decay coefficient (non-negative).
    eps : float
        Numerical floor (non-negative).
    t : int
        1-based timestep for bias correction.
    theta : array-like, optional
        Current parameters; default zeros.

    Returns
    -------
    result : RichResult
        Keys: theta, step, adam_step, decay_step, m, v, estimate, n, method.

    Examples
    --------
    >>> r = geron_adamw([1.0], theta=[2.0], eta=0.1, wd=0.5, t=1)
    >>> round(float(r["adam_step"][0]), 6), round(float(r["decay_step"][0]), 6)
    (-0.1, -0.1)
    >>> round(float(r["theta"][0]), 6)
    1.8
    >>> r0 = geron_adamw([1.0], theta=[2.0], eta=0.1, wd=0.0, t=1)
    >>> round(float(r0["theta"][0]), 6)
    1.9

    References
    ----------
    Géron Ch 11
    """
    g = np.atleast_1d(np.asarray(grads, dtype=float))
    if g.size == 0:
        raise ValueError("geron_adamw: grads is empty")
    if not np.all(np.isfinite(g)):
        raise ValueError("geron_adamw: grads contains non-finite values")
    mm = np.zeros_like(g) if m is None else np.atleast_1d(np.asarray(m, dtype=float))
    vv = np.zeros_like(g) if v is None else np.atleast_1d(np.asarray(v, dtype=float))
    th = np.zeros_like(g) if theta is None else np.atleast_1d(np.asarray(theta, dtype=float))
    for name, arr in (("m", mm), ("v", vv), ("theta", th)):
        if arr.shape != g.shape:
            raise ValueError(f"geron_adamw: {name} has shape {arr.shape} but grads has shape {g.shape}")
    beta1, beta2, lr, decay, e = float(b1), float(b2), float(eta), float(wd), float(eps)
    if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
        raise ValueError(f"geron_adamw: b1 and b2 must lie in [0, 1), got b1={beta1}, b2={beta2}")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("geron_adamw: eta must be a positive finite learning rate")
    if decay < 0:
        raise ValueError("geron_adamw: wd must be non-negative")
    if e < 0:
        raise ValueError("geron_adamw: eps must be non-negative")
    step_t = int(t)
    if step_t < 1:
        raise ValueError("geron_adamw: t must be a 1-based timestep >= 1")
    if np.any(vv < 0):
        raise ValueError("geron_adamw: v must be non-negative (it accumulates squared gradients)")

    m_new = beta1 * mm + (1.0 - beta1) * g
    v_new = beta2 * vv + (1.0 - beta2) * g * g
    m_hat = m_new / (1.0 - beta1**step_t)
    v_hat = v_new / (1.0 - beta2**step_t)
    adam_step = -lr * m_hat / (np.sqrt(v_hat) + e)
    decay_step = -lr * decay * th
    step = adam_step + decay_step
    theta_next = th + step

    return RichResult(
        title="AdamW step",
        summary_lines=[("Timestep", step_t), ("Weight decay", decay), ("Step L2 norm", float(np.linalg.norm(step)))],
        payload={
            "theta": theta_next,
            "theta_next": theta_next,
            "step": step,
            "adam_step": adam_step,
            "decay_step": decay_step,
            "m": m_new,
            "v": v_new,
            "m_hat": m_hat,
            "v_hat": v_hat,
            "t": step_t,
            "estimate": float(np.linalg.norm(step)),
            "n": int(g.size),
            "method": "AdamW (Adam with decoupled weight decay)",
        },
    )


def cheatsheet():
    return "hmadmw: AdamW: decoupled weight decay"


# compact alias per ledger/NAMING.md
geronadamw = geron_adamw
