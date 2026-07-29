# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AdaGrad update: per-parameter learning rate scaled by accumulated squared gradients."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_adagrad_update"]

_METHOD = "AdaGrad parameter update"


def geron_adagrad_update(theta, grad, s, eta, eps=1e-10):
    r"""One AdaGrad step.

    .. math::
        s \leftarrow s + g \odot g, \qquad
        \theta \leftarrow \theta - \eta\, g \oslash (\sqrt{s} + \epsilon)

    The accumulator :math:`s` only grows, so the effective learning rate
    decays monotonically per coordinate -- which is exactly why AdaGrad
    stalls on deep nets and RMSProp replaces the sum with a moving
    average.

    Parameters
    ----------
    theta : array-like
        Current parameters.
    grad : array-like
        Gradient at ``theta``, same shape.
    s : array-like
        Running sum of squared gradients, same shape, non-negative.
    eta : float
        Base learning rate, positive.
    eps : float, optional
        Numerical floor, default ``1e-10``.

    Returns
    -------
    RichResult
        Payload keys ``theta_new``, ``s_new``, ``effective_lr``,
        ``step``, ``step_norm``, ``estimate`` (L2 norm of the update),
        ``n``, ``method``.

    References
    ----------
    Géron Ch 11, AdaGrad section.

    Examples
    --------
    >>> r = geron_adagrad_update([1.0], [2.0], [0.0], 0.1, eps=0.0)
    >>> r["s_new"]
    [4.0]
    >>> round(r["theta_new"][0], 10)
    0.9
    """
    theta = np.asarray(theta, dtype=float)
    grad = np.asarray(grad, dtype=float)
    s = np.asarray(s, dtype=float)
    if theta.shape != grad.shape:
        raise ValueError(f"theta shape {theta.shape} != grad shape {grad.shape}.")
    if theta.shape != s.shape:
        raise ValueError(f"theta shape {theta.shape} != s shape {s.shape}.")
    if theta.size == 0:
        raise ValueError("theta is empty.")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(grad)):
        raise ValueError("theta and grad must be finite.")
    if np.any(s < 0):
        raise ValueError("s (accumulated squared gradients) must be non-negative.")
    eta = float(eta)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"eta must be a positive finite float, got {eta}.")
    eps = float(eps)
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}.")

    s_new = s + grad * grad
    denom = np.sqrt(s_new) + eps
    if np.any(denom == 0):
        raise ValueError(
            "zero denominator: a coordinate has zero accumulated gradient and "
            "eps=0; pass eps > 0."
        )
    eff_lr = eta / denom
    step = eff_lr * grad
    theta_new = theta - step

    return RichResult(
        title="AdaGrad update",
        summary_lines=[("Step L2 norm", float(np.linalg.norm(step)))],
        payload={
            "theta_new": theta_new.tolist(),
            "s_new": s_new.tolist(),
            "effective_lr": eff_lr.tolist(),
            "step": step.tolist(),
            "step_norm": float(np.linalg.norm(step)),
            "estimate": float(np.linalg.norm(step)),
            "n": int(theta.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grada2: AdaGrad -- s += g^2; theta -= eta*g/(sqrt(s)+eps)"
