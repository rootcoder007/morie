# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exponential linear unit."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_elu"]


def geron_elu(z, alpha=1.0):
    """
    Exponential linear unit.

    Formula: ELU(z) = z if z>=0 else alpha*(exp(z)-1)

    The derivative is returned alongside the activation: 1 on the positive
    side and ``a + alpha`` on the negative side, so the function is C^1 at
    the origin exactly when ``alpha == 1``, which is why that is the
    default. The saturation floor ``-alpha`` is reported too.

    Parameters
    ----------
    z : array-like
        Pre-activations.
    alpha : float, default 1.0
        Negative-side scale; must be positive and finite.

    Returns
    -------
    result : RichResult
        Keys: a, activation, derivative, saturation, is_c1, estimate,
        n, method.

    Examples
    --------
    >>> r = geron_elu([-1.0, 0.0, 2.0])
    >>> [round(float(v), 9) for v in r["a"]]
    [-0.632120559, 0.0, 2.0]
    >>> [round(float(v), 9) for v in r["derivative"]]
    [0.367879441, 1.0, 1.0]
    >>> geron_elu([-100.0], alpha=2.0)["saturation"]
    -2.0
    >>> round(float(geron_elu([-100.0], alpha=2.0)["a"][0]), 9)
    -2.0

    References
    ----------
    Géron Ch 11
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if z.size == 0:
        raise ValueError("geron_elu: z is empty")
    if not np.all(np.isfinite(z)):
        raise ValueError("geron_elu: z contains non-finite values")
    a_scale = float(alpha)
    if not np.isfinite(a_scale) or a_scale <= 0:
        raise ValueError(f"geron_elu: alpha must be positive and finite, got {alpha!r}")

    pos = z >= 0
    a = np.where(pos, z, a_scale * (np.expm1(np.minimum(z, 0.0))))
    d = np.where(pos, 1.0, a + a_scale)

    return RichResult(
        title="ELU activation",
        summary_lines=[("alpha", a_scale), ("Saturation floor", -a_scale)],
        payload={
            "a": a,
            "activation": a,
            "derivative": d,
            "saturation": float(-a_scale),
            "alpha": a_scale,
            "is_c1": bool(a_scale == 1.0),
            "estimate": float(np.mean(a)),
            "n": int(z.size),
            "method": "ELU(z) = z if z >= 0 else alpha*(exp(z) - 1)",
        },
    )


def cheatsheet():
    return "hmelu: Exponential linear unit"
