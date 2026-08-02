# morie.fn -- function file (rootcoder007/morie)
"""Softmax, the gradient of log-sum-exp -- Boyd & Vandenberghe Sec. 3.1.5."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_lse"]


def boyd_lse(x, temperature=1.0):
    r"""The softmax :math:`\nabla \operatorname{lse}(x)_i =
    e^{x_i}/\sum_j e^{x_j}`.

    Invariant to adding a constant to every entry -- softmax(x + c) =
    softmax(x) -- which is both a modelling fact (only DIFFERENCES of
    logits are identified) and the licence for the shift-by-max trick that
    prevents overflow.

    The temperature interpolates between two limits: as
    :math:`T \to 0` the softmax approaches a one-hot at the argmax, and as
    :math:`T \to \infty` it approaches the uniform distribution. Neither
    limit is attainable in floating point, and the function says which one
    it is near.

    Parameters
    ----------
    x : array-like
        Logits.
    temperature : float
        Positive scaling; the softmax is applied to ``x / temperature``.

    Returns
    -------
    RichResult
        ``value`` (the softmax), ``entropy``, ``argmax``, ``max_prob``,
        ``jacobian_diag``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    >>> import numpy as np
    >>> p = boyd_lse([1.0, 2.0, 3.0])["value"]
    >>> bool(abs(p.sum() - 1) < 1e-12 and np.all(p > 0))
    True

    Shift invariance: adding a constant to every logit changes nothing.

    >>> q = boyd_lse([101.0, 102.0, 103.0])["value"]
    >>> bool(np.max(np.abs(p - q)) < 1e-12)
    True

    Low temperature concentrates on the argmax; high temperature flattens
    toward uniform.

    >>> bool(boyd_lse([1.0, 2.0, 3.0], 0.05)["max_prob"] > 0.99)
    True
    >>> bool(boyd_lse([1.0, 2.0, 3.0], 100.0)["entropy"] > 1.09)
    True

    >>> boyd_lse([1.0], temperature=0.0)
    Traceback (most recent call last):
        ...
    ValueError: temperature must be positive
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if xv.size == 0:
        raise ValueError("x must be non-empty")
    t = float(temperature)
    if t <= 0:
        raise ValueError("temperature must be positive")
    z = xv / t
    z = z - z.max()
    e = np.exp(z)
    p = e / e.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = float(-np.sum(np.where(p > 0, p * np.log(p), 0.0)))
    return RichResult(
        title="Softmax",
        summary_lines=[("n", int(xv.size)), ("temperature", t),
                       ("max prob", float(p.max())), ("entropy", ent)],
        payload={
            "value": p, "entropy": ent, "argmax": int(np.argmax(xv)),
            "max_prob": float(p.max()),
            "jacobian_diag": p * (1.0 - p) / t,
            "temperature": t, "method": "boyd_lse",
        },
    )


def cheatsheet():
    return "cvxlse: shift invariant, so only logit DIFFERENCES are identified; T->0 one-hot, T->inf uniform"
