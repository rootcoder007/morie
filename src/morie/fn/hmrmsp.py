# morie.fn -- function file (rootcoder007/morie)
"""RMSProp: exponentially weighted gradient-squared average."""

from . import _array_core as np

from ._richresult import RichResult
from .grrmsp import geron_rmsprop_update

__all__ = ["rmsprop", "geron_rmsprop"]


def rmsprop(grads, params=None, lr=0.001, rho=0.9, eps=1e-7):
    r"""Run RMSProp over a sequence of gradients.

    Accepts a whole gradient TRAJECTORY rather than a single step, so
    the decay of the accumulator is visible: pass ``grads`` of shape
    ``(T, p)`` and get the parameter path and the effective learning
    rate at every step.

    That trajectory is the thing worth looking at. The effective rate
    :math:`\eta/\sqrt{s+\epsilon}` should FALL where gradients are
    consistently large and RECOVER where they go quiet -- the recovery
    is precisely what AdaGrad cannot do, and seeing it is the check
    that the decay is wired up.

    Parameters
    ----------
    grads : array-like, shape (T, p) or (p,)
    params : array-like, shape (p,), optional
        Starting point. Zeros by default.
    lr, rho, eps : float

    Returns
    -------
    RichResult
        ``params`` (final), ``path`` (T by p), ``effective_lr`` (T by p),
        ``state``.

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 11.
    Tieleman and Hinton (2012).

    Examples
    --------
    >>> out = rmsprop([[1.0], [1.0], [1.0]], lr=0.1)
    >>> out["path"].shape
    (3, 1)
    """
    G = np.atleast_2d(np.asarray(grads, dtype=float))
    if G.ndim != 2:
        raise ValueError("grads must be 1- or 2-dimensional.")
    T, p = G.shape
    theta = np.zeros(p) if params is None else np.asarray(
        params, dtype=float
    ).ravel().copy()
    if theta.size != p:
        raise ValueError(
            "params has %d entries for %d gradient components."
            % (theta.size, p)
        )
    state = None
    path = np.empty((T, p))
    effl = np.empty((T, p))
    for t in range(T):
        out = geron_rmsprop_update(theta, G[t], state, lr, rho, eps)
        theta = out["params"]
        state = out["state"]
        path[t] = theta
        effl[t] = out["effective_lr"]
    return RichResult(
        payload={
            "estimate": theta,
            "params": theta,
            "path": path,
            "effective_lr": effl,
            "state": state,
            "note": (
                "the effective rate falls under sustained gradients and "
                "recovers when they go quiet; AdaGrad's summed accumulator "
                "cannot recover, which is why it stalls"
            ),
            "n_steps": int(T),
            "method": "RMSProp over a gradient trajectory",
        }
    )


def cheatsheet():
    return "hmrmsp: RMSProp across a gradient path, exposing the effective rate"


#: Catalogue alias for :func:`rmsprop`.
geron_rmsprop = rmsprop


# compact alias per ledger/NAMING.md
geronrmsprop = geron_rmsprop
