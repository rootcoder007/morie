# morie.fn -- function file (rootcoder007/morie)
"""RMSProp -- Tieleman & Hinton (2012)."""

from __future__ import annotations

from . import _array_core as np

from ._optim import as_vector, init_state, step_result

__all__ = ["rmsprop"]


def rmsprop(g, rho=0.9, lr=0.001, eps=1e-8, state=None):
    r"""One RMSProp step from the gradient ``g``.

    RMSProp replaces AdaGrad's ever-growing sum with an exponentially decayed
    average of squared gradients,

    .. math::
        v_t = \rho\, v_{t-1} + (1-\rho)\, g_t^2, \qquad
        \Delta\theta_t = -\,\alpha\, g_t / (\sqrt{v_t} + \epsilon),

    so the effective learning rate can recover instead of decaying to zero.
    Unlike :func:`~morie.fn.adamopt.adam` there is no momentum term and no
    bias correction, so the first few steps are larger than the asymptotic
    ones.

    Parameters
    ----------
    g : array-like
        Gradient at the current parameters.
    rho : float
        Decay rate for the squared-gradient average, in [0, 1).
    lr : float
        Step size.
    eps : float
        Denominator floor.
    state : dict, optional
        ``state`` from the previous call.

    Returns
    -------
    RichResult
        ``update``, ``state``, the running average ``v``, and ``step_norm``.

    References
    ----------
    Tieleman, T., & Hinton, G. (2012). Lecture 6.5 -- RMSProp. *COURSERA:
        Neural Networks for Machine Learning*.

    Examples
    --------
    >>> import numpy as np
    >>> x, st = np.zeros(1), None
    >>> for _ in range(5000):
    ...     r = rmsprop(2 * (x - 3.0), lr=0.05, state=st)
    ...     x, st = x + r["update"], r["state"]
    >>> bool(abs(x[0] - 3.0) < 1e-2)
    True

    On a constant gradient the running average converges, so unlike AdaGrad
    the step size settles rather than shrinking away.

    >>> st = None
    >>> for _ in range(500):
    ...     r = rmsprop([1.0], lr=0.1, state=st)
    ...     st = r["state"]
    >>> bool(abs(abs(r["update"][0]) - 0.1) < 1e-3)
    True
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    g = as_vector(g)
    st = init_state(state, g.size, keys=("v",))
    st["v"] = rho * st["v"] + (1.0 - rho) * g**2
    update = -lr * g / (np.sqrt(st["v"]) + eps)
    return step_result(update, st, "RMSProp", v=st["v"])


def cheatsheet():
    return "rmsoptm: RMSProp -- decayed mean of squared gradients; no momentum, no bias correction"
