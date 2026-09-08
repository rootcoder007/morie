# morie.fn -- function file (rootcoder007/morie)
"""Nesterov accelerated gradient -- Nesterov (1983), Sutskever et al. (2013)."""

from __future__ import annotations

from ._optim import as_vector, init_state, step_result

__all__ = ["nesterov_accelerated"]


def nesterov_accelerated(g, mu=0.9, lr=0.01, state=None):
    r"""One Nesterov accelerated-gradient step.

    In the Sutskever et al. formulation the velocity is updated as for
    classical momentum but the parameter step looks *ahead* along it:

    .. math::
        v_t = \mu v_{t-1} - \alpha g_t, \qquad
        \Delta\theta_t = \mu v_t - \alpha g_t.

    The extra :math:`\mu(v_t - v_{t-1})` relative to heavy-ball is the
    correction that gives the accelerated :math:`O(1/t^2)` rate on smooth
    convex problems, and in practice damps the overshoot momentum produces
    near a minimum.

    ``g`` should be evaluated at the look-ahead point
    :math:`\theta + \mu v_{t-1}` for the exact method; evaluating it at
    :math:`\theta` gives the common approximate variant.

    Parameters
    ----------
    g : array-like
        Gradient (see note above on where to evaluate it).
    mu : float
        Momentum coefficient in [0, 1).
    lr : float
        Step size.
    state : dict, optional
        ``state`` from the previous call.

    Returns
    -------
    RichResult
        ``update``, ``state``, ``v``.

    References
    ----------
    Nesterov, Y. (1983). A method of solving a convex programming problem
        with convergence rate O(1/k^2). *Soviet Mathematics Doklady*, 27(2),
        372-376.
    Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the
        importance of initialization and momentum in deep learning.
        *ICML 2013*, 1139-1147.

    Examples
    --------
    >>> import numpy as np
    >>> x, st = np.zeros(1), None
    >>> for _ in range(500):
    ...     r = nesterov_accelerated(2 * (x - 3.0), mu=0.9, lr=0.01, state=st)
    ...     x, st = x + r["update"], r["state"]
    >>> bool(abs(x[0] - 3.0) < 1e-6)
    True

    The first step is larger than heavy-ball's by the look-ahead term:

    >>> float(round(nesterov_accelerated([1.0], mu=0.9, lr=0.1)["update"][0], 6))
    -0.19
    """
    if not 0.0 <= mu < 1.0:
        raise ValueError("mu must be in [0, 1)")
    g = as_vector(g)
    st = init_state(state, g.size, keys=("v",))
    st["v"] = mu * st["v"] - lr * g
    update = mu * st["v"] - lr * g
    return step_result(update, st, "Nesterov accelerated gradient", v=st["v"])


def cheatsheet():
    return "nesterv: NAG -- update looks ahead (mu*v - lr*g); evaluate g at theta + mu*v for the exact form"
