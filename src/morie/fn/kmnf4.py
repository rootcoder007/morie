# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NormalFloat4 (NF4): the quantile grid for normally distributed
weights."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_nf4_datatype", "normal_quantile"]


def normal_quantile(u, tol=1e-14, max_iter=200):
    """Phi^{-1}(u) by bisection on math.erf -- stdlib only, accurate to
    ``tol``. Refuses u outside (0, 1), where the quantile is infinite."""
    u = float(u)
    if not 0.0 < u < 1.0:
        raise ValueError(
            f"Phi^-1({u}) is infinite; the argument must lie strictly "
            "inside (0, 1).")
    lo, hi = -40.0, 40.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < u:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def kamath_nf4_datatype(n_bins=16):
    """q_i = Phi^{-1}((i + 0.5) / n_bins) for i = 0 .. n_bins - 1.

    The equal-mass quantile grid of the standard normal: with 16 bins
    it is the 4-bit NormalFloat type, information-theoretically
    optimal for N(0, 1) weights. ``normalized`` rescales the grid to
    [-1, 1] as the datatype is stored in practice; both are returned
    so nothing has to be recomputed, and neither is silently
    substituted for the other.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, NF4 (Dettmers et al.
    2023).

    Examples
    --------
    >>> out = kamath_nf4_datatype()
    >>> len(out["levels"])
    16
    >>> abs(out["levels"][7] + out["levels"][8]) < 1e-12
    True
    >>> round(out["levels"][8], 6)
    0.078412
    >>> out["normalized"][-1]
    1.0
    >>> two = kamath_nf4_datatype(2)
    >>> abs(two["levels"][1] - 0.6744897501960817) < 1e-9
    True
    """
    n = int(n_bins)
    if n < 2:
        raise ValueError(f"a data type needs at least 2 levels; got {n}.")
    levels = np.array([normal_quantile((i + 0.5) / n) for i in range(n)])
    m = float(np.max(np.abs(levels)))
    if m == 0:
        raise ValueError("the quantile grid collapsed to zero.")
    normalized = levels / m
    widths = np.diff(levels)
    return RichResult(payload={
        "levels": [float(v) for v in levels],
        "normalized": [float(v) for v in normalized],
        "bin_widths": [float(v) for v in widths],
        "n_bits": math.log2(n) if (n & (n - 1)) == 0 else None,
        "estimate": float(levels[-1]), "n": n,
        "method": "NF4 equal-mass normal quantile grid"})


def cheatsheet():
    return "kmnf4: q_i = Phi^-1((i+0.5)/n), plus the [-1,1] normalised grid"
