# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Swish / SiLU activation."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsigm import geron_sigmoid

__all__ = ["geron_swish"]


def geron_swish(z, beta=1.0):
    """
    Swish / SiLU activation.

    Formula: swish(z) = z * sigmoid(z)

    The sigmoid factor is delegated to :func:`morie.fn.hmsigm.geron_sigmoid`
    rather than recomputed, so the overflow-safe branch is shared. The
    derivative follows from the product rule:
    ``swish'(z) = s + beta*z*s*(1-s)`` with ``s = sigmoid(beta*z)``.
    Unlike ReLU, swish is non-monotonic: it dips to a finite minimum near
    ``z = -1.278`` for ``beta = 1``.

    Parameters
    ----------
    z : array-like
        Pre-activation values. Must be finite.
    beta : float, default 1.0
        Slope of the gate; ``beta = 1`` is SiLU. Must be finite.

    Returns
    -------
    result : RichResult
        Keys: a, grad, gate, estimate, n, method.

    Examples
    --------
    >>> r = geron_swish([-1.0, 0.0, 1.0])
    >>> [round(float(v), 6) for v in r["a"]]
    [-0.268941, 0.0, 0.731059]
    >>> round(float(r["grad"][1]), 6)
    0.5
    >>> bool(min(r["a"]) < 0.0)
    True

    References
    ----------
    Géron Ch 11
    """
    x = np.atleast_1d(np.asarray(z, dtype=float))
    if x.size == 0:
        raise ValueError("geron_swish: z is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("geron_swish: z contains non-finite values")
    b = float(beta)
    if not np.isfinite(b):
        raise ValueError("geron_swish: beta must be finite")

    s = np.asarray(geron_sigmoid(b * x)["a"], dtype=float)
    a = x * s
    grad = s + b * x * s * (1.0 - s)

    return RichResult(
        title="Swish / SiLU activation",
        summary_lines=[("Mean activation", float(np.mean(a))), ("beta", b)],
        interpretation="Swish is smooth and non-monotonic; it keeps a small negative lobe that ReLU discards.",
        payload={
            "a": a,
            "grad": grad,
            "gate": s,
            "beta": b,
            "estimate": float(np.mean(a)),
            "n": int(x.size),
            "method": "Swish z*sigmoid(beta z) with sigmoid delegated to hmsigm",
        },
    )


def cheatsheet():
    return "hmswi: Swish / SiLU activation"
