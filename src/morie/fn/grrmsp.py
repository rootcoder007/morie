# morie.fn -- function file (rootcoder007/morie)
"""RMSProp optimiser update."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_rmsprop_update"]


def geron_rmsprop_update(params, grads, state=None, lr=0.001, rho=0.9,
                         eps=1e-7, steps=1):
    r"""One or more RMSProp steps.

    .. math::
       s \leftarrow \rho s + (1-\rho)\, g \otimes g, \qquad
       \theta \leftarrow \theta - \frac{\eta\, g}{\sqrt{s + \epsilon}}

    RMSProp is AdaGrad with the accumulation replaced by an
    exponentially weighted average, and that single change is the
    point. AdaGrad sums squared gradients forever, so the effective
    learning rate decays monotonically and the optimiser stalls before
    reaching the optimum on anything but a simple convex problem.
    Decaying the average lets the scale forget old gradients, so a
    parameter whose gradient was large early can still move later.

    ``eps`` sits INSIDE the square root here, matching the formulation
    in Geron; some implementations put it outside. The difference is
    negligible except when ``s`` is near zero, which is exactly the
    first step, where the outside form takes a much larger stride.

    Parameters
    ----------
    params, grads : array-like
        Current parameters and their gradients.
    state : dict, optional
        Carries ``s`` between calls.
    lr, rho, eps : float
    steps : int
        Repeat the update this many times with the same gradient. Only
        meaningful for inspecting the trajectory.

    Returns
    -------
    RichResult
        ``params``, ``state``, ``step_size``, ``effective_lr``.

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 11,
    RMSProp. Hinton, Coursera lecture 6e (2012).
    Tieleman and Hinton (2012).

    Examples
    --------
    >>> out = geron_rmsprop_update([1.0], [1.0], lr=0.1)
    >>> round(float(out["params"][0]), 4)
    0.6838
    """
    p = np.asarray(params, dtype=float).copy()
    g = np.asarray(grads, dtype=float)
    if g.shape != p.shape:
        raise ValueError(
            "params and grads must match in shape, got %s and %s."
            % (p.shape, g.shape)
        )
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1), got %r." % rho)
    if lr <= 0:
        raise ValueError("lr must be positive, got %r." % lr)
    s = np.zeros_like(p) if state is None else np.asarray(
        state.get("s", np.zeros_like(p)), dtype=float
    ).copy()
    step = np.zeros_like(p)
    for _ in range(max(int(steps), 1)):
        s = rho * s + (1.0 - rho) * g * g
        step = lr * g / np.sqrt(s + eps)
        p = p - step
    return RichResult(
        payload={
            "estimate": p,
            "params": p,
            "state": {"s": s},
            "step_size": step,
            "effective_lr": lr / np.sqrt(s + eps),
            "note": (
                "the exponential average is what separates RMSProp from "
                "AdaGrad: a summed accumulator decays the learning rate "
                "monotonically and stalls, a decayed one can forget"
            ),
            "eps_placement": "inside the square root, as in Geron ch. 11",
            "lr": float(lr),
            "rho": float(rho),
            "method": "RMSProp update",
        }
    )


def cheatsheet():
    return "grrmsp: RMSProp step with the decayed squared-gradient scale"
