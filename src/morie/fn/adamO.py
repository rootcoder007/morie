# morie.fn -- function file (rootcoder007/morie)
"""Adam update applied to a parameter vector."""

from __future__ import annotations

from ._richresult import RichResult
from .adamopt import adam

__all__ = ["adam_optimizer"]


def adam_optimizer(theta, grad, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, state=None):
    r"""Adam, expressed as a parameter-in / parameter-out step.

    Thin front-end over :func:`~morie.fn.adamopt.adam` for callers that would
    rather hand over the parameters than apply the increment themselves. The
    arithmetic is identical; only the calling convention differs.

    Parameters
    ----------
    theta : array-like
        Current parameters.
    grad : array-like
        Gradient at ``theta``; must match its size.
    lr, beta1, beta2, eps : float
        As for :func:`~morie.fn.adamopt.adam`.
    state : dict, optional
        ``state`` from the previous call.

    Returns
    -------
    RichResult
        ``theta`` (updated), ``update``, ``state``, ``m``, ``v``.

    References
    ----------
    Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic
        optimization. *ICLR 2015*. arXiv:1412.6980.

    Examples
    --------
    >>> import numpy as np
    >>> th, st = np.zeros(2), None
    >>> for _ in range(4000):
    ...     r = adam_optimizer(th, 2 * (th - np.array([1.0, -2.0])), lr=0.05, state=st)
    ...     th, st = r["theta"], r["state"]
    >>> [float(round(v, 3)) for v in th]
    [1.0, -2.0]

    >>> adam_optimizer([0.0, 0.0], [1.0])
    Traceback (most recent call last):
        ...
    ValueError: grad has 1 entries but theta has 2
    """
    from morie.fn import _array_core as np

    th = np.atleast_1d(np.asarray(theta, dtype=float)).ravel()
    gr = np.atleast_1d(np.asarray(grad, dtype=float)).ravel()
    if gr.size != th.size:
        raise ValueError(f"grad has {gr.size} entries but theta has {th.size}")
    r = adam(gr, beta1=beta1, beta2=beta2, lr=lr, eps=eps, state=state)
    return RichResult(
        title="Adam optimizer step",
        summary_lines=[("step", int(r["t"])), ("|update|", float(r["step_norm"]))],
        payload={
            "theta": th + r["update"],
            "update": r["update"],
            "state": r["state"],
            "m": r["m"],
            "v": r["v"],
            "t": int(r["t"]),
            "step_norm": float(r["step_norm"]),
            "method": "adam_optimizer",
        },
    )


def cheatsheet():
    return "adamO: parameter-in/parameter-out Adam; same math as adamopt.adam, different calling convention"


# compact alias per ledger/NAMING.md
adamoptimizer = adam_optimizer
