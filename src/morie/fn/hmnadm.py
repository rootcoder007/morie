# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NAdam: Adam with Nesterov momentum."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_nadam"]


def geron_nadam(grads, m=None, v=None, b1=0.9, b2=0.999, eta=0.001, t=1, eps=1e-8, theta=None):
    """
    NAdam: Adam with Nesterov momentum.

    Formula: m_hat = Nesterov-corrected first moment; update as in Adam

    The moments are accumulated exactly as in
    :func:`~morie.fn.hmadam.geron_adam`; the only change is the
    correction applied to the first moment, which mixes the CURRENT
    gradient with the decayed momentum,

        m_hat = b1 * m_t / (1 - b1^(t+1)) + (1 - b1) * g_t / (1 - b1^t),

    so the step is taken from where momentum is about to land rather than
    from where it is -- the Nesterov look-ahead, applied without ever
    evaluating the gradient twice.

    Parameters
    ----------
    grads : array-like
        Gradient at the current parameters.
    m, v : array-like, optional
        Moment accumulators; default zeros.
    b1, b2 : float
        Decay rates in [0, 1).
    eta : float
        Learning rate (positive).
    t : int
        1-based timestep.
    eps : float
        Numerical floor (non-negative).
    theta : array-like, optional
        Current parameters; default zeros.

    Returns
    -------
    result : RichResult
        Keys: theta, step, m, v, m_hat, v_hat, estimate, n, method.

    Examples
    --------
    First step from zero moments with g = 1, eta = 0.1: m = 0.1,
    v = 0.001, v_hat = 1, and
    m_hat = 0.9*0.1/(1-0.9^2) + 0.1*1/(1-0.9) = 0.09/0.19 + 1.

    >>> r = geron_nadam([1.0], eta=0.1, t=1)
    >>> round(float(r["m"][0]), 12), round(float(r["v"][0]), 12)
    (0.1, 0.001)
    >>> round(float(r["m_hat"][0]), 9)
    1.473684211
    >>> round(float(r["step"][0]), 6)
    -0.147368

    A sign flip in the gradient flips the step:

    >>> round(float(geron_nadam([-1.0], eta=0.1, t=1)["step"][0]), 6)
    0.147368

    References
    ----------
    Geron Ch 11
    """
    g = np.atleast_1d(np.asarray(grads, dtype=float))
    if g.size == 0:
        raise ValueError("geron_nadam: grads is empty")
    if not np.all(np.isfinite(g)):
        raise ValueError("geron_nadam: grads contains non-finite values")
    mm = np.zeros_like(g) if m is None else np.atleast_1d(np.asarray(m, dtype=float))
    vv = np.zeros_like(g) if v is None else np.atleast_1d(np.asarray(v, dtype=float))
    th = np.zeros_like(g) if theta is None else np.atleast_1d(np.asarray(theta, dtype=float))
    for name, arr in (("m", mm), ("v", vv), ("theta", th)):
        if arr.shape != g.shape:
            raise ValueError(f"geron_nadam: {name} has shape {arr.shape} but grads has shape {g.shape}")
    beta1, beta2, lr, e = float(b1), float(b2), float(eta), float(eps)
    if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
        raise ValueError(f"geron_nadam: b1 and b2 must lie in [0, 1), got b1={beta1}, b2={beta2}")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("geron_nadam: eta must be a positive finite learning rate")
    if e < 0:
        raise ValueError("geron_nadam: eps must be non-negative")
    step_t = int(t)
    if step_t < 1:
        raise ValueError("geron_nadam: t must be a 1-based timestep >= 1")
    if np.any(vv < 0):
        raise ValueError("geron_nadam: v must be non-negative (it accumulates squared gradients)")

    m_new = beta1 * mm + (1.0 - beta1) * g
    v_new = beta2 * vv + (1.0 - beta2) * g * g
    m_hat = beta1 * m_new / (1.0 - beta1 ** (step_t + 1)) + (1.0 - beta1) * g / (1.0 - beta1**step_t)
    v_hat = v_new / (1.0 - beta2**step_t)
    step = -lr * m_hat / (np.sqrt(v_hat) + e)
    theta_next = th + step
    return RichResult(
        title="NAdam step",
        summary_lines=[("Timestep", step_t), ("Step L2 norm", float(np.linalg.norm(step)))],
        interpretation="Nesterov correction pulls the current gradient into the step without a second evaluation.",
        payload={
            "theta": theta_next,
            "theta_next": theta_next,
            "step": step,
            "m": m_new,
            "v": v_new,
            "m_hat": m_hat,
            "v_hat": v_hat,
            "t": step_t,
            "estimate": float(np.linalg.norm(step)),
            "n": int(g.size),
            "method": "NAdam (Adam moments with a Nesterov-corrected first moment)",
        },
    )


def cheatsheet():
    return "hmnadm: NAdam, Adam with Nesterov momentum"
