# morie.fn -- function file (rootcoder007/morie)
"""AdaGrad -- Duchi, Hazan & Singer (2011)."""

from __future__ import annotations

import numpy as np

from ._optim import as_vector, init_state, step_result

__all__ = ["adagrad"]


def adagrad(g, lr=0.01, eps=1e-8, state=None):
    r"""One AdaGrad step from the gradient ``g``.

    AdaGrad accumulates the *un-decayed* sum of squared gradients,
    :math:`G_t = \sum_{s \le t} g_s^2`, and divides the step by its root:

    .. math::
        \Delta\theta_t = -\,\alpha\, g_t / (\sqrt{G_t} + \epsilon).

    Coordinates with a long history of large gradients therefore take
    smaller steps. Because nothing decays, the effective learning rate is
    monotonically non-increasing and eventually stalls -- that behaviour is
    the method, not a defect, and it is why RMSProp introduced a decay.

    Parameters
    ----------
    g : array-like
        Gradient at the current parameters.
    lr : float
        Base step size.
    eps : float
        Denominator floor.
    state : dict, optional
        ``state`` from the previous call; ``None`` starts fresh.

    Returns
    -------
    RichResult
        ``update``, ``state``, the accumulator ``G``, and ``step_norm``.

    References
    ----------
    Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods
        for online learning and stochastic optimization. *JMLR*, 12,
        2121-2159.

    Examples
    --------
    >>> import numpy as np
    >>> x, st = np.zeros(1), None
    >>> for _ in range(20000):
    ...     r = adagrad(2 * (x - 3.0), lr=0.5, state=st)
    ...     x, st = x + r["update"], r["state"]
    >>> bool(abs(x[0] - 3.0) < 1e-2)
    True

    The accumulator only grows, so successive steps on a constant gradient
    shrink.

    >>> r1 = adagrad([1.0], lr=0.1)
    >>> r2 = adagrad([1.0], lr=0.1, state=r1["state"])
    >>> bool(abs(r2["update"][0]) < abs(r1["update"][0]))
    True
    """
    g = as_vector(g)
    st = init_state(state, g.size, keys=("G",))
    st["G"] = st["G"] + g**2
    update = -lr * g / (np.sqrt(st["G"]) + eps)
    return step_result(update, st, "AdaGrad", G=st["G"])


def cheatsheet():
    return "adgrad: AdaGrad -- undecayed sum of squared gradients; step size decays monotonically"
