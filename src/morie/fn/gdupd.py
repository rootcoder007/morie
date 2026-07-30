# morie.fn -- function file (rootcoder007/morie)
"""Gradient-descent parameter update."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["gradient_descent_update"]


def gradient_descent_update(beta, grad, alpha=0.01):
    r"""Apply one gradient-descent update to the parameter vector.

    .. math::
        \beta_{t+1} = \beta_t - \alpha \nabla f(\beta_t).

    This is the whole of batch gradient descent; everything else in the
    family (momentum, Adam, RMSProp) replaces the raw gradient with some
    filtered version of it.

    Parameters
    ----------
    beta : array-like
        Current parameters.
    grad : array-like
        Gradient at ``beta``; must match its shape.
    alpha : float
        Learning rate. Must be positive.

    Returns
    -------
    RichResult
        ``beta`` (updated), ``update`` (the increment applied), and
        ``step_norm``.

    References
    ----------
    Cauchy, A. (1847). Methode generale pour la resolution des systemes
        d'equations simultanees. *Comptes Rendus de l'Academie des Sciences*,
        25, 536-538.

    Examples
    --------
    >>> import numpy as np
    >>> b = np.zeros(2)
    >>> for _ in range(2000):
    ...     b = gradient_descent_update(b, 2 * (b - np.array([1.0, -2.0])), 0.05)["beta"]
    >>> [float(round(v, 6)) for v in b]
    [1.0, -2.0]

    >>> gradient_descent_update([1.0], [0.5], -0.1)
    Traceback (most recent call last):
        ...
    ValueError: alpha must be positive
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    grad = np.atleast_1d(np.asarray(grad, dtype=float))
    if beta.shape != grad.shape:
        raise ValueError(f"beta {beta.shape} and grad {grad.shape} must have the same shape")
    update = -alpha * grad
    new = beta + update
    return RichResult(
        title="Gradient descent update",
        summary_lines=[("alpha", float(alpha)), ("|update|", float(np.linalg.norm(update)))],
        payload={
            "beta": new,
            "update": update,
            "step_norm": float(np.linalg.norm(update)),
            "alpha": float(alpha),
            "method": "gradient_descent_update",
        },
    )


def cheatsheet():
    return "gdupd: beta <- beta - alpha*grad; the base every adaptive rule filters the gradient for"
