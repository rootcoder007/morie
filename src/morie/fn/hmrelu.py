# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rectified linear unit activation."""

from . import _array_core as np

from ._richresult import RichResult
from .relu import relu as _relu

__all__ = ["geron_relu"]


def geron_relu(z, leaky=0.0):
    """
    Rectified linear unit activation.

    Formula: ReLU(z) = max(0, z)

    The activation and its subgradient are DELEGATED to the finished
    implementation :func:`morie.fn.relu.relu`; this wrapper adds the dead
    fraction, which is the diagnostic Geron uses to spot dying ReLUs (a
    unit whose input is negative for every instance gets no gradient and
    never recovers). Pass ``leaky`` to keep a slope on the negative side.
    The derivative at exactly 0 is taken to be ``leaky``; ReLU is not
    differentiable there and any convention is a choice.

    Parameters
    ----------
    z : array-like
        Pre-activations.
    leaky : float, default 0.0
        Negative slope; 0 is the plain ReLU.

    Returns
    -------
    result : RichResult
        Keys: a, gradient, dead_fraction, estimate, n, method.

    Examples
    --------
    >>> r = geron_relu([-2.0, 0.0, 3.0])
    >>> [float(v) for v in r["a"]]
    [0.0, 0.0, 3.0]
    >>> [float(v) for v in r["gradient"]]
    [0.0, 0.0, 1.0]
    >>> round(float(r["dead_fraction"]), 6)
    0.666667
    >>> [float(v) for v in geron_relu([-2.0, 3.0], leaky=0.5)["a"]]
    [-1.0, 3.0]

    References
    ----------
    Geron Ch 9
    """
    a = np.atleast_1d(np.asarray(z, dtype=float))
    if a.size == 0:
        raise ValueError("geron_relu: z is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("geron_relu: z contains non-finite values")
    slope = float(leaky)
    if slope < 0:
        raise ValueError(f"geron_relu: leaky slope must be >= 0, got {slope}")

    res = _relu(a, leaky=slope)
    # +0.0 normalises the -0.0 that slope*negative produces when slope is 0.
    out = np.asarray(res.extra["output"], dtype=float) + 0.0
    grad = np.asarray(res.extra["gradient"], dtype=float)
    dead = float(np.mean(a <= 0))
    return RichResult(
        title="ReLU activation" if slope == 0 else f"Leaky ReLU (alpha={slope})",
        summary_lines=[("Active fraction", 1.0 - dead), ("Mean activation", float(np.mean(out)))],
        interpretation="A unit dead for every instance in the batch gets zero gradient; leaky ReLU avoids that.",
        payload={
            "a": out,
            "output": out,
            "gradient": grad,
            "dead_fraction": dead,
            "leaky": slope,
            "estimate": out,
            "n": int(a.size),
            "method": "ReLU delegated to morie.fn.relu.relu",
        },
    )


def cheatsheet():
    return "hmrelu: Rectified linear unit activation"
