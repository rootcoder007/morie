# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adam optimizer: momentum + RMSProp with bias correction."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_adam"]


def geron_adam(grads, m=None, v=None, b1=0.9, b2=0.999, eta=0.001, eps=1e-8, t=1, theta=None):
    """
    Adam optimizer: momentum + RMSProp with bias correction.

    Formula: m_hat = m/(1-b1^t); v_hat = v/(1-b2^t);
    theta <- theta - eta*m_hat/(sqrt(v_hat)+eps)

    One Adam step. The moment estimates are updated first
    (m <- b1*m + (1-b1)*g, v <- b2*v + (1-b2)*g^2), then bias-corrected
    with the timestep `t`, which must be 1 on the first call.

    Parameters
    ----------
    grads : array-like
        Gradient of the loss w.r.t. the parameters.
    m, v : array-like, optional
        First and second moment accumulators from the previous step;
        default zeros.
    b1, b2 : float
        Exponential decay rates in [0, 1).
    eta : float
        Learning rate (positive).
    eps : float
        Numerical floor (non-negative).
    t : int
        1-based timestep used for bias correction.
    theta : array-like, optional
        Current parameters; default zeros. Returned updated as `theta_next`.

    Returns
    -------
    result : RichResult
        Keys: theta, step, m, v, m_hat, v_hat, estimate, n, method.

    Examples
    --------
    >>> r = geron_adam([1.0], m=[0.0], v=[0.0], eta=0.1, t=1)
    >>> round(float(r["m"][0]), 12), round(float(r["v"][0]), 12)
    (0.1, 0.001)
    >>> round(float(r["m_hat"][0]), 9), round(float(r["v_hat"][0]), 9)
    (1.0, 1.0)
    >>> round(float(r["step"][0]), 6)
    -0.1
    >>> r2 = geron_adam([-2.0], eta=0.5, t=1)
    >>> round(float(r2["step"][0]), 6)
    0.5

    References
    ----------
    Géron Ch 11
    """
    g = np.atleast_1d(np.asarray(grads, dtype=float))
    if g.size == 0:
        raise ValueError("geron_adam: grads is empty")
    if not np.all(np.isfinite(g)):
        raise ValueError("geron_adam: grads contains non-finite values")
    mm = np.zeros_like(g) if m is None else np.atleast_1d(np.asarray(m, dtype=float))
    vv = np.zeros_like(g) if v is None else np.atleast_1d(np.asarray(v, dtype=float))
    th = np.zeros_like(g) if theta is None else np.atleast_1d(np.asarray(theta, dtype=float))
    for name, arr in (("m", mm), ("v", vv), ("theta", th)):
        if arr.shape != g.shape:
            raise ValueError(f"geron_adam: {name} has shape {arr.shape} but grads has shape {g.shape}")
    beta1, beta2, lr, e = float(b1), float(b2), float(eta), float(eps)
    if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
        raise ValueError(f"geron_adam: b1 and b2 must lie in [0, 1), got b1={beta1}, b2={beta2}")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("geron_adam: eta must be a positive finite learning rate")
    if e < 0:
        raise ValueError("geron_adam: eps must be non-negative")
    step_t = int(t)
    if step_t < 1:
        raise ValueError("geron_adam: t must be a 1-based timestep >= 1")
    if np.any(vv < 0):
        raise ValueError("geron_adam: v must be non-negative (it accumulates squared gradients)")

    m_new = beta1 * mm + (1.0 - beta1) * g
    v_new = beta2 * vv + (1.0 - beta2) * g * g
    m_hat = m_new / (1.0 - beta1**step_t)
    v_hat = v_new / (1.0 - beta2**step_t)
    step = -lr * m_hat / (np.sqrt(v_hat) + e)
    theta_next = th + step

    return RichResult(
        title="Adam step",
        summary_lines=[("Timestep", step_t), ("Step L2 norm", float(np.linalg.norm(step)))],
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
            "method": "Adam (bias-corrected momentum + RMSProp)",
        },
    )


def cheatsheet():
    return "hmadam: Adam optimizer: momentum + RMSProp with bias correction"


# compact alias per ledger/NAMING.md
geronadam = geron_adam
