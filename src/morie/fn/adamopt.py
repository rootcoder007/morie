# morie.fn -- function file (rootcoder007/morie)
"""Adam optimiser -- Kingma & Ba (2015)."""

from __future__ import annotations

from . import _array_core as np

from ._optim import as_vector, init_state, step_result

__all__ = ["adam"]


def adam(g, beta1=0.9, beta2=0.999, lr=1e-3, eps=1e-8, state=None):
    r"""One Adam step from the gradient ``g``.

    Adam keeps exponential moving averages of the gradient and of its square,

    .. math::
        m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
        v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,

    corrects both for their initialisation at zero,
    :math:`\hat m_t = m_t / (1 - \beta_1^t)` and likewise for
    :math:`\hat v_t`, and steps by

    .. math::
        \Delta\theta_t = -\,\alpha\, \hat m_t / (\sqrt{\hat v_t} + \epsilon).

    The bias correction is what makes the first few steps the right size; it
    is not optional decoration.

    Parameters
    ----------
    g : array-like
        Gradient at the current parameters. Any shape; flattened to 1-D.
    beta1, beta2 : float
        Decay rates for the first and second moment, both in [0, 1).
    lr : float
        Step size :math:`\alpha`.
    eps : float
        Denominator floor, for numerical stability only.
    state : dict, optional
        The ``state`` returned by the previous call. ``None`` starts a fresh
        run with zeroed moments.

    Returns
    -------
    RichResult
        ``update`` (the increment to *add* to the parameters), ``state`` to
        feed the next call, plus ``m``, ``v``, ``t`` and ``step_norm``.

    References
    ----------
    Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic
        optimization. *ICLR 2015*. arXiv:1412.6980.

    Examples
    --------
    Minimise ``f(x) = (x - 3)^2`` from ``x = 0``; the iterate moves toward 3.

    >>> import numpy as np
    >>> x, st = np.zeros(1), None
    >>> for _ in range(4000):
    ...     r = adam(2 * (x - 3.0), lr=0.05, state=st)
    ...     x, st = x + r["update"], r["state"]
    >>> bool(abs(x[0] - 3.0) < 1e-3)
    True

    Bias correction makes the very first step almost exactly ``lr`` in size,
    whatever the gradient's scale.

    >>> float(np.round(adam([1000.0], lr=0.01)["update"][0], 6))
    -0.01
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
    update = -lr * m_hat / (np.sqrt(v_hat) + eps)
    return step_result(update, st, "Adam", m=st["m"], v=st["v"], m_hat=m_hat, v_hat=v_hat)


def cheatsheet():
    return "adamopt: Adam -- bias-corrected first/second moments; thread `state` between steps"
