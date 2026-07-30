# morie.fn -- function file (rootcoder007/morie)
"""AdamW step in gradient-in / update-out form -- Loshchilov & Hutter (2019).

See :func:`~morie.fn.adamw.adamw` for the weights-in / weights-out variant
that also carries its own moment buffers; the arithmetic agrees.
"""

from __future__ import annotations

import numpy as np

from ._optim import as_vector, init_state, step_result

__all__ = ["adamw_step"]


def adamw_step(g, beta1=0.9, beta2=0.999, lr=1e-3, wd=0.01, eps=1e-8, theta=None, state=None):
    r"""One AdamW step: Adam, with weight decay applied outside the adaptive scaling.

    Adam with L2 regularisation folds :math:`\lambda\theta` into the gradient,
    so the decay is then divided by :math:`\sqrt{\hat v_t}` along with
    everything else -- parameters with large gradient history get decayed
    *less*, which is not what regularisation is supposed to do. AdamW keeps
    the two separate:

    .. math::
        \Delta\theta_t = -\,\alpha\left(
            \hat m_t / (\sqrt{\hat v_t} + \epsilon) + \lambda \theta_t
        \right).

    That decoupling is the entire content of the method.

    Parameters
    ----------
    g : array-like
        Gradient of the *unregularised* loss.
    beta1, beta2 : float
        Moment decay rates, in [0, 1).
    lr : float
        Step size.
    wd : float
        Decoupled weight-decay coefficient :math:`\lambda`.
    eps : float
        Denominator floor.
    theta : array-like, optional
        Current parameters, needed for the decay term. If omitted the decay
        is skipped and the step reduces to plain Adam.
    state : dict, optional
        ``state`` from the previous call.

    Returns
    -------
    RichResult
        ``update``, ``state``, ``m``, ``v``, and ``decay`` (the decay
        component of the step, for diagnostics).

    References
    ----------
    Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay
        regularization. *ICLR 2019*. arXiv:1711.05101.

    Examples
    --------
    With zero gradient the step is pure decay, pulling the parameter toward
    zero by exactly ``lr * wd * theta``.

    >>> float(round(adamw_step([0.0], lr=0.1, wd=0.5, theta=[2.0])["update"][0], 6))
    -0.1

    Omitting ``theta`` gives the same step as :func:`~morie.fn.adamopt.adam`.

    >>> from morie.fn.adamopt import adam
    >>> a = adamw_step([1.0], lr=0.01)["update"][0]
    >>> b = adam([1.0], lr=0.01)["update"][0]
    >>> bool(abs(a - b) < 1e-12)
    True

    >>> import numpy as np
    >>> x, st = np.zeros(1), None
    >>> for _ in range(4000):
    ...     r = adamw_step(2 * (x - 3.0), lr=0.05, wd=0.0, state=st)
    ...     x, st = x + r["update"], r["state"]
    >>> bool(abs(x[0] - 3.0) < 1e-3)
    True
    """
    if not 0.0 <= beta1 < 1.0:
        raise ValueError("beta1 must be in [0, 1)")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("beta2 must be in [0, 1)")
    g = as_vector(g)
    st = init_state(state, g.size, keys=("m", "v"))
    t = st["t"]
    st["m"] = beta1 * st["m"] + (1.0 - beta1) * g
    st["v"] = beta2 * st["v"] + (1.0 - beta2) * g**2
    m_hat = st["m"] / (1.0 - beta1**t)
    v_hat = st["v"] / (1.0 - beta2**t)
    adaptive = -lr * m_hat / (np.sqrt(v_hat) + eps)
    if theta is None:
        decay = np.zeros_like(g)
    else:
        th = np.atleast_1d(np.asarray(theta, dtype=float)).ravel()
        if th.size != g.size:
            raise ValueError(f"theta has {th.size} entries but g has {g.size}")
        decay = -lr * wd * th
    return step_result(adaptive + decay, st, "AdamW", m=st["m"], v=st["v"], decay=decay)


def cheatsheet():
    return "adwopt: AdamW -- decay applied outside the 1/sqrt(v) scaling; pass `theta` or it is plain Adam"
