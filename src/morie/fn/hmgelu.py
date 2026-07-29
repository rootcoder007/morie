# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian error linear unit (GELU)."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_gelu"]

_SQRT2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _phi(z):
    """Standard normal CDF, via math.erf (no scipy)."""
    return 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(z / _SQRT2))


def _phi_pdf(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def geron_gelu(z, approximate=False):
    """
    Gaussian error linear unit (GELU).

    Formula: GELU(z) = z * Phi(z)

    ``Phi`` is the standard normal CDF, evaluated exactly through
    ``math.erf``. The derivative ``Phi(z) + z phi(z)`` is returned as
    well; it dips below zero for ``z`` around -1, which is the non-
    monotonicity that distinguishes GELU from ReLU. The tanh
    approximation Géron mentions is available via ``approximate=True``
    and its gap from the exact value is always reported.

    Parameters
    ----------
    z : array-like
        Pre-activations.
    approximate : bool, default False
        Return the tanh approximation
        ``0.5 z (1 + tanh(sqrt(2/pi)(z + 0.044715 z^3)))`` as ``a``.

    Returns
    -------
    result : RichResult
        Keys: a, activation, exact, approx, derivative, max_abs_gap,
        estimate, n, method.

    Examples
    --------
    >>> r = geron_gelu([-1.0, 0.0, 1.0, 2.0])
    >>> [round(float(v), 9) for v in r["a"]]
    [-0.158655254, 0.0, 0.841344746, 1.954499736]
    >>> [round(float(v), 9) for v in r["derivative"]]
    [-0.083315471, 0.5, 1.083315471, 1.085231801]
    >>> round(float(geron_gelu([1.0], approximate=True)["a"][0]), 6)
    0.841192
    >>> round(float(geron_gelu([1.0])["max_abs_gap"]), 6)
    0.000153

    References
    ----------
    Géron Ch 11
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if z.size == 0:
        raise ValueError("geron_gelu: z is empty")
    if not np.all(np.isfinite(z)):
        raise ValueError("geron_gelu: z contains non-finite values")

    cdf = _phi(z)
    exact = z * cdf
    approx = 0.5 * z * (1.0 + np.tanh(_SQRT_2_OVER_PI * (z + 0.044715 * z**3)))
    deriv = cdf + z * _phi_pdf(z)
    a = approx if approximate else exact

    return RichResult(
        title="GELU activation",
        summary_lines=[("Mode", "tanh approximation" if approximate else "exact erf")],
        interpretation="GELU is non-monotone: its derivative is negative near z = -1.",
        payload={
            "a": a,
            "activation": a,
            "exact": exact,
            "approx": approx,
            "derivative": deriv,
            "max_abs_gap": float(np.max(np.abs(exact - approx))),
            "estimate": float(np.mean(a)),
            "n": int(z.size),
            "method": "GELU(z) = z * Phi(z)" + (" (tanh approximation)" if approximate else ""),
        },
    )


def cheatsheet():
    return "hmgelu: Gaussian error linear unit (GELU)"
