# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adam optimizer step (bias-corrected first and second moments)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_adam_update"]

_METHOD = "Adam optimizer step with bias correction"


def geron_adam_update(theta, grad, m, s, t, eta, b1=0.9, b2=0.999, eps=1e-8):
    r"""One Adam step, bias correction included.

    .. math::
        m &\leftarrow \beta_1 m + (1-\beta_1) g \\
        s &\leftarrow \beta_2 s + (1-\beta_2) g \odot g \\
        \hat m &= m / (1 - \beta_1^{t}), \qquad
        \hat s = s / (1 - \beta_2^{t}) \\
        \theta &\leftarrow \theta - \eta\, \hat m \oslash (\sqrt{\hat s} + \epsilon)

    The bias correction is not decoration.  At ``t=1`` from zeroed
    moments, :math:`m = (1-\beta_1) g` is a hundredth of the gradient;
    dividing by :math:`1-\beta_1^1 = 1-\beta_1` restores it exactly, so
    the first step has magnitude ``eta`` rather than ``eta/100``.

    Parameters
    ----------
    theta, grad, m, s : array-like
        Parameters, gradient, first-moment and second-moment
        accumulators. All four must share a shape.
    t : int
        1-based step counter. Must be at least 1 -- ``t=0`` makes the
        bias correction divide by zero.
    eta : float
        Learning rate, positive.
    b1, b2 : float, optional
        Exponential decay rates in ``[0, 1)``.
    eps : float, optional
        Numerical floor, default ``1e-8``.

    Returns
    -------
    RichResult
        Payload keys ``theta_new``, ``m_new``, ``s_new``, ``m_hat``,
        ``s_hat``, ``step``, ``estimate`` (L2 norm of the update),
        ``n``, ``method``.

    References
    ----------
    Géron Ch 11, Eq 11-8 (Adam).

    Examples
    --------
    First step from zeroed moments has size ``eta`` regardless of the
    gradient scale -- that is the bias correction working:

    >>> r = geron_adam_update([1.0], [0.1], [0.0], [0.0], t=1, eta=0.001)
    >>> round(r["m_hat"][0], 10)
    0.1
    >>> round(r["s_hat"][0], 10)
    0.01
    >>> round(r["theta_new"][0], 8)
    0.999
    """
    theta = np.asarray(theta, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    s = np.asarray(s, dtype=float)
    for name, arr in (("grad", grad), ("m", m), ("s", s)):
        if arr.shape != theta.shape:
            raise ValueError(f"{name} shape {arr.shape} != theta shape {theta.shape}.")
    if theta.size == 0:
        raise ValueError("theta is empty.")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(grad)):
        raise ValueError("theta and grad must be finite.")
    if np.any(s < 0):
        raise ValueError("s (second moment) must be non-negative.")
    t = int(t)
    if t < 1:
        raise ValueError(
            f"t is the 1-based step counter and must be >= 1 (bias correction "
            f"divides by 1 - beta^t), got {t}."
        )
    eta = float(eta)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"eta must be a positive finite float, got {eta}.")
    b1 = float(b1)
    b2 = float(b2)
    for name, b in (("b1", b1), ("b2", b2)):
        if not (0.0 <= b < 1.0):
            raise ValueError(f"{name} must lie in [0, 1), got {b}.")
    eps = float(eps)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")

    m_new = b1 * m + (1.0 - b1) * grad
    s_new = b2 * s + (1.0 - b2) * grad * grad
    m_hat = m_new / (1.0 - b1**t)
    s_hat = s_new / (1.0 - b2**t)
    step = eta * m_hat / (np.sqrt(s_hat) + eps)
    theta_new = theta - step

    return RichResult(
        title="Adam update",
        summary_lines=[("Step", int(t)), ("Step L2 norm", float(np.linalg.norm(step)))],
        payload={
            "theta_new": theta_new.tolist(),
            "m_new": m_new.tolist(),
            "s_new": s_new.tolist(),
            "m_hat": m_hat.tolist(),
            "s_hat": s_hat.tolist(),
            "step": step.tolist(),
            "t": t,
            "estimate": float(np.linalg.norm(step)),
            "n": int(theta.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "gradmo: Adam -- bias-corrected first/second moments, theta -= eta*m_hat/(sqrt(s_hat)+eps)"
