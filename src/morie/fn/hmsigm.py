# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Logistic sigmoid activation function."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_sigmoid"]


def geron_sigmoid(t):
    """
    Logistic sigmoid activation function.

    Formula: sigma(t) = 1 / (1 + exp(-t))

    Evaluated by the overflow-safe branch (``exp(t)/(1+exp(t))`` for
    negative inputs) so large-magnitude arguments saturate to 0 or 1
    instead of raising. The derivative ``sigma*(1-sigma)`` is returned
    alongside, which is what backpropagation consumes.

    Parameters
    ----------
    t : array-like
        Pre-activation values (logits). Must be finite.

    Returns
    -------
    result : RichResult
        Keys: a, grad, estimate, n, method.

    Examples
    --------
    >>> r = geron_sigmoid([-1.0, 0.0, 1.0])
    >>> [round(float(v), 6) for v in r["a"]]
    [0.268941, 0.5, 0.731059]
    >>> round(float(r["grad"][1]), 6)
    0.25
    >>> float(geron_sigmoid([-1000.0])["a"][0])
    0.0

    References
    ----------
    Géron Ch 4
    """
    z = np.atleast_1d(np.asarray(t, dtype=float))
    if z.size == 0:
        raise ValueError("geron_sigmoid: t is empty")
    if not np.all(np.isfinite(z)):
        raise ValueError("geron_sigmoid: t contains non-finite values")

    pos = z >= 0
    a = np.empty_like(z)
    a[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    a[~pos] = ez / (1.0 + ez)
    grad = a * (1.0 - a)

    return RichResult(
        title="Logistic sigmoid",
        summary_lines=[("Mean activation", float(np.mean(a))), ("Max slope", float(np.max(grad)))],
        interpretation="sigma maps the real line onto (0, 1); its slope peaks at 0.25 when t = 0.",
        payload={
            "a": a,
            "sigma": a,
            "grad": grad,
            "estimate": float(np.mean(a)),
            "n": int(z.size),
            "method": "Logistic sigmoid, overflow-safe branch, with elementwise derivative",
        },
    )


def cheatsheet():
    return "hmsigm: Logistic sigmoid activation function"
