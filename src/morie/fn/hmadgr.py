# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AdaGrad: per-parameter learning rates scaled by historical gradients."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_adagrad"]


def geron_adagrad(grads, s=None, eta=0.01, eps=1e-10, theta=None):
    """
    AdaGrad: per-parameter learning rates scaled by historical gradients.

    Formula: s <- s + g^2; theta <- theta - eta * g / (sqrt(s) + eps)

    Parameters
    ----------
    grads : array-like
        Gradient of the loss w.r.t. the parameters.
    s : array-like, optional
        Accumulator of squared gradients; default zeros. Must be
        non-negative and shaped like `grads`.
    eta : float
        Base learning rate (positive).
    eps : float
        Numerical floor (non-negative).
    theta : array-like, optional
        Current parameters; default zeros.

    Returns
    -------
    result : RichResult
        Keys: theta, step, s, effective_lr, estimate, n, method.

    Examples
    --------
    >>> r = geron_adagrad([2.0], s=[0.0], eta=0.1, eps=0.0)
    >>> float(r["s"][0])
    4.0
    >>> round(float(r["step"][0]), 12)
    -0.1
    >>> r2 = geron_adagrad([2.0], s=[4.0], eta=0.1, eps=0.0)
    >>> float(r2["s"][0]), round(float(r2["step"][0]), 6)
    (8.0, -0.070711)
    >>> round(float(r2["effective_lr"][0]), 6)
    0.035355

    References
    ----------
    Géron Ch 11
    """
    g = np.atleast_1d(np.asarray(grads, dtype=float))
    if g.size == 0:
        raise ValueError("geron_adagrad: grads is empty")
    if not np.all(np.isfinite(g)):
        raise ValueError("geron_adagrad: grads contains non-finite values")
    ss = np.zeros_like(g) if s is None else np.atleast_1d(np.asarray(s, dtype=float))
    th = np.zeros_like(g) if theta is None else np.atleast_1d(np.asarray(theta, dtype=float))
    if ss.shape != g.shape:
        raise ValueError(f"geron_adagrad: s has shape {ss.shape} but grads has shape {g.shape}")
    if th.shape != g.shape:
        raise ValueError(f"geron_adagrad: theta has shape {th.shape} but grads has shape {g.shape}")
    if np.any(ss < 0):
        raise ValueError("geron_adagrad: s must be non-negative (it accumulates squared gradients)")
    lr, e = float(eta), float(eps)
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("geron_adagrad: eta must be a positive finite learning rate")
    if e < 0:
        raise ValueError("geron_adagrad: eps must be non-negative")

    s_new = ss + g * g
    denom = np.sqrt(s_new) + e
    if np.any(denom == 0):
        # Only reachable when a coordinate has never seen a gradient and eps=0;
        # that coordinate has no defined scale, so refuse rather than emit inf.
        raise ValueError("geron_adagrad: zero accumulator with eps=0 leaves the per-parameter scale undefined")
    step = -lr * g / denom
    theta_next = th + step

    return RichResult(
        title="AdaGrad step",
        summary_lines=[("Step L2 norm", float(np.linalg.norm(step)))],
        interpretation="Per-parameter rates shrink monotonically because s only accumulates.",
        payload={
            "theta": theta_next,
            "theta_next": theta_next,
            "step": step,
            "s": s_new,
            "effective_lr": lr / denom,
            "estimate": float(np.linalg.norm(step)),
            "n": int(g.size),
            "method": "AdaGrad (per-parameter rates from accumulated squared gradients)",
        },
    )


def cheatsheet():
    return "hmadgr: AdaGrad: per-parameter learning rates scaled by historical gradients"


# compact alias per ledger/NAMING.md
geronadagrad = geron_adagrad
