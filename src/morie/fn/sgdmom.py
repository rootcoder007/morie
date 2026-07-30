# morie.fn -- function file (rootcoder007/morie)
"""SGD with classical (heavy-ball) momentum -- Polyak (1964)."""

from __future__ import annotations

from ._optim import as_vector, init_state, step_result

__all__ = ["sgd_momentum"]


def sgd_momentum(g, mu=0.9, lr=0.01, state=None):
    r"""One heavy-ball momentum step from the gradient ``g``.

    The velocity accumulates past gradients geometrically,

    .. math::
        v_t = \mu v_{t-1} - \alpha g_t, \qquad \Delta\theta_t = v_t.

    On a constant gradient the step approaches :math:`-\alpha g/(1-\mu)`, so
    momentum multiplies the effective learning rate by :math:`1/(1-\mu)` --
    the reason raising ``mu`` without lowering ``lr`` diverges.

    Parameters
    ----------
    g : array-like
        Gradient at the current parameters.
    mu : float
        Momentum coefficient in [0, 1). ``mu=0`` is plain SGD.
    lr : float
        Step size.
    state : dict, optional
        ``state`` from the previous call.

    Returns
    -------
    RichResult
        ``update`` (equal to the new velocity), ``state``, ``v``.

    References
    ----------
    Polyak, B. T. (1964). Some methods of speeding up the convergence of
        iteration methods. *USSR Computational Mathematics and Mathematical
        Physics*, 4(5), 1-17.

    Examples
    --------
    >>> import numpy as np
    >>> x, st = np.zeros(1), None
    >>> for _ in range(500):
    ...     r = sgd_momentum(2 * (x - 3.0), mu=0.9, lr=0.01, state=st)
    ...     x, st = x + r["update"], r["state"]
    >>> bool(abs(x[0] - 3.0) < 1e-6)
    True

    The 1/(1-mu) amplification on a constant gradient:

    >>> st = None
    >>> for _ in range(400):
    ...     r = sgd_momentum([1.0], mu=0.9, lr=0.01, state=st)
    ...     st = r["state"]
    >>> bool(abs(r["update"][0] + 0.1) < 1e-6)
    True
    """
    if not 0.0 <= mu < 1.0:
        raise ValueError("mu must be in [0, 1)")
    g = as_vector(g)
    st = init_state(state, g.size, keys=("v",))
    st["v"] = mu * st["v"] - lr * g
    return step_result(st["v"].copy(), st, "SGD with momentum", v=st["v"])


def cheatsheet():
    return "sgdmom: heavy-ball momentum; steady-state step is -lr*g/(1-mu), so raise mu and lower lr together"
