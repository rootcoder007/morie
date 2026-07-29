# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mish activation: z * tanh(softplus(z))."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_mish"]

_METHOD = "Mish activation"


def _softplus(z):
    # log1p(exp(z)) overflows for large z; max(z,0) + log1p(exp(-|z|)) does not.
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def geron_mish(z):
    """
    Mish activation: z * tanh(softplus(z)).

    Formula: Mish(z) = z * tanh(ln(1 + exp(z)))

    Smooth and non-monotonic: like Swish it dips slightly below zero for
    small negative inputs before flattening, and unlike ReLU it is
    differentiable everywhere.  The softplus is evaluated as
    ``max(z,0) + log1p(exp(-|z|))`` so that large inputs do not overflow
    ``exp``; the naive ``log(1+exp(z))`` returns ``inf`` around z=710 and
    would poison the product.

    The analytic derivative is returned as well:
    ``d/dz = tanh(sp) + z * sigma(z) * (1 - tanh(sp)^2)`` with
    ``sp = softplus(z)``.

    Parameters
    ----------
    z : array-like
        Pre-activations.

    Returns
    -------
    result : RichResult
        Keys: activation, derivative, softplus, minimum, estimate, n, method.

    Examples
    --------
    ``Mish(0) = 0 * tanh(ln 2) = 0`` exactly:

    >>> float(geron_mish([0.0])["activation"][0])
    0.0

    Large positive inputs are essentially the identity, and there is no
    overflow at 800:

    >>> r = geron_mish([20.0, 800.0])
    >>> [round(float(v), 6) for v in r["activation"]]
    [20.0, 800.0]

    The derivative agrees with a central difference at z = -1:

    >>> h = 1e-6
    >>> up = float(geron_mish([-1.0 + h])["activation"][0])
    >>> dn = float(geron_mish([-1.0 - h])["activation"][0])
    >>> analytic = float(geron_mish([-1.0])["derivative"][0])
    >>> bool(abs((up - dn) / (2 * h) - analytic) < 1e-7)
    True

    Mish is non-monotonic, so its minimum is strictly negative:

    >>> bool(geron_mish([-1.0])["activation"][0] < 0)
    True

    References
    ----------
    Géron Ch 11
    """
    a = np.atleast_1d(np.asarray(z, dtype=float))
    if a.size == 0:
        raise ValueError("geron_mish: z is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("geron_mish: z contains non-finite values")

    sp = _softplus(a)
    th = np.tanh(sp)
    out = a * th
    deriv = th + a * _sigmoid(a) * (1.0 - th * th)

    return RichResult(
        title="Mish activation",
        summary_lines=[("Units", int(a.size)), ("Minimum output", float(np.min(out)))],
        interpretation="Smooth and non-monotonic: a small negative dip keeps gradients alive below zero.",
        payload={
            "activation": out,
            "derivative": deriv,
            "softplus": sp,
            "minimum": float(np.min(out)),
            "estimate": float(np.mean(out)),
            "n": int(a.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmish: Mish z*tanh(softplus(z)) with overflow-safe softplus and analytic derivative"
