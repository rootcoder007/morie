# morie.fn -- function file (rootcoder007/morie)
"""RMSProp update applied to a parameter vector."""

from __future__ import annotations

from ._richresult import RichResult
from .rmsoptm import rmsprop

__all__ = ["rmsprop_optimizer"]


def rmsprop_optimizer(theta, grad, lr=0.001, rho=0.9, eps=1e-8, state=None):
    r"""RMSProp, expressed as a parameter-in / parameter-out step.

    Thin front-end over :func:`~morie.fn.rmsoptm.rmsprop`; identical
    arithmetic, different calling convention.

    Parameters
    ----------
    theta : array-like
        Current parameters.
    grad : array-like
        Gradient at ``theta``; must match its size.
    lr, rho, eps : float
        As for :func:`~morie.fn.rmsoptm.rmsprop`.
    state : dict, optional
        ``state`` from the previous call.

    Returns
    -------
    RichResult
        ``theta`` (updated), ``update``, ``state``, ``v``.

    References
    ----------
    Tieleman, T., & Hinton, G. (2012). Lecture 6.5 -- RMSProp. *COURSERA:
        Neural Networks for Machine Learning*.

    Examples
    --------
    >>> import numpy as np
    >>> th, st = np.zeros(1), None
    >>> for _ in range(5000):
    ...     r = rmsprop_optimizer(th, 2 * (th - 3.0), lr=0.05, state=st)
    ...     th, st = r["theta"], r["state"]
    >>> bool(abs(th[0] - 3.0) < 1e-2)
    True
    """
    import numpy as np

    th = np.atleast_1d(np.asarray(theta, dtype=float)).ravel()
    gr = np.atleast_1d(np.asarray(grad, dtype=float)).ravel()
    if gr.size != th.size:
        raise ValueError(f"grad has {gr.size} entries but theta has {th.size}")
    r = rmsprop(gr, rho=rho, lr=lr, eps=eps, state=state)
    return RichResult(
        title="RMSProp optimizer step",
        summary_lines=[("step", int(r["t"])), ("|update|", float(r["step_norm"]))],
        payload={
            "theta": th + r["update"],
            "update": r["update"],
            "state": r["state"],
            "v": r["v"],
            "t": int(r["t"]),
            "step_norm": float(r["step_norm"]),
            "method": "rmsprop_optimizer",
        },
    )


def cheatsheet():
    return "rmspO: parameter-in/parameter-out RMSProp; same math as rmsoptm.rmsprop"
