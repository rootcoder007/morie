# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hyperbolic tangent activation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_tanh"]


def geron_tanh(z):
    """
    Hyperbolic tangent activation.

    Formula: tanh(z) = (e^z - e^-z) / (e^z + e^-z)

    Computed with the stable library kernel (algebraically identical to
    ``2*sigmoid(2z) - 1``) and returned with its derivative
    ``1 - tanh(z)^2``, which is what backpropagation through an RNN or a
    tanh MLP consumes.

    Parameters
    ----------
    z : array-like
        Pre-activation values. Must be finite.

    Returns
    -------
    result : RichResult
        Keys: a, grad, estimate, n, method.

    Examples
    --------
    >>> r = geron_tanh([-1.0, 0.0, 1.0])
    >>> [round(float(v), 6) for v in r["a"]]
    [-0.761594, 0.0, 0.761594]
    >>> float(r["grad"][1])
    1.0
    >>> round(float(geron_tanh([2.0])["grad"][0]), 6)
    0.070651

    References
    ----------
    Géron Ch 9
    """
    x = np.atleast_1d(np.asarray(z, dtype=float))
    if x.size == 0:
        raise ValueError("geron_tanh: z is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("geron_tanh: z contains non-finite values")

    a = np.tanh(x)
    grad = 1.0 - a * a

    return RichResult(
        title="Hyperbolic tangent",
        summary_lines=[("Mean activation", float(np.mean(a))), ("Max slope", float(np.max(grad)))],
        interpretation="tanh is zero-centred on (-1, 1); its slope is 1 at the origin and decays either side.",
        payload={
            "a": a,
            "grad": grad,
            "estimate": float(np.mean(a)),
            "n": int(x.size),
            "method": "Hyperbolic tangent activation with elementwise derivative 1 - tanh^2",
        },
    )


def cheatsheet():
    return "hmtanh: Hyperbolic tangent activation"
