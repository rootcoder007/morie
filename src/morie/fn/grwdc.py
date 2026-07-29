# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AdamW: decoupled weight decay applied directly to the parameters."""

import numpy as np

from ._richresult import RichResult
from .gradmo import geron_adam_update

__all__ = ["geron_adamw_decoupled_weight_decay"]

_METHOD = "AdamW (decoupled weight decay)"


def geron_adamw_decoupled_weight_decay(theta, grad, m, s, t, eta, b1=0.9, b2=0.999,
                                       eps=1e-8, lam=0.01):
    r"""Adam step plus a decay term that never touches the moments.

    .. math::
        \theta \leftarrow \theta - \eta\left(
            \frac{\hat m}{\sqrt{\hat s} + \epsilon} + \lambda\theta\right)

    L2 regularisation and weight decay are the same thing for plain SGD
    and *not* the same thing for Adam.  Folding :math:`\lambda\theta` into
    the gradient sends it through the second-moment normalisation, so
    parameters with large gradient history get decayed less -- exactly
    backwards.  AdamW applies the decay after the normalisation, which is
    the one-line difference this module exists to make explicit.  The
    Adam half is delegated to :func:`morie.fn.gradmo.geron_adam_update`
    (bias correction included there).

    Parameters
    ----------
    theta, grad, m, s : array-like
        Same shape.
    t : int
        1-based step counter.
    eta : float
        Positive learning rate.
    b1, b2 : float, optional
    eps : float, optional
    lam : float, optional
        Non-negative decay coefficient.

    Returns
    -------
    RichResult
        Payload keys ``theta_new``, ``m_new``, ``s_new``, ``adam_step``,
        ``decay_step``, ``step``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 11, AdamW section.

    Examples
    --------
    First step from zeroed moments: the Adam part has size ``eta``, and
    the decay subtracts ``eta * lam * theta`` on top.

    >>> r = geron_adamw_decoupled_weight_decay([1.0], [0.1], [0.0], [0.0], t=1,
    ...                                        eta=0.001, lam=0.1)
    >>> round(r["adam_step"][0], 8)
    0.001
    >>> round(r["decay_step"][0], 8)
    0.0001
    >>> round(r["theta_new"][0], 8)
    0.9989

    With ``lam = 0`` it is exactly Adam:

    >>> a = geron_adamw_decoupled_weight_decay([1.0], [0.1], [0.0], [0.0], 1, 0.001, lam=0.0)
    >>> round(a["theta_new"][0], 8)
    0.999
    """
    inner = geron_adam_update(theta, grad, m, s, t, eta, b1=b1, b2=b2, eps=eps)
    th = np.asarray(theta, dtype=float)
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError(f"lam must be finite and non-negative, got {lam}.")

    adam_step = np.asarray(inner["step"], dtype=float).reshape(th.shape)
    decay_step = float(eta) * lam * th
    step = adam_step + decay_step
    new = th - step

    return RichResult(
        title="AdamW update",
        summary_lines=[("Step", int(t)), ("Decay L2 norm", float(np.linalg.norm(decay_step)))],
        payload={
            "theta_new": new.tolist(),
            "m_new": inner["m_new"],
            "s_new": inner["s_new"],
            "adam_step": adam_step.tolist(),
            "decay_step": decay_step.tolist(),
            "step": step.tolist(),
            "t": int(t),
            "estimate": float(np.linalg.norm(step)),
            "n": int(th.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grwdc: theta -= eta(m_hat/(sqrt(s_hat)+eps) + lam*theta); decay OUTSIDE the normalisation"
