# morie.fn -- function file (rootcoder007/morie)
"""Getis-Ord global G statistic."""

from __future__ import annotations

import math

from . import _stats_core as stats
from . import _t4core as T

from ._richresult import RichResult

__all__ = ["getis_ord_g"]


def getis_ord_g(x, W):
    """Getis-Ord global G, with its exact randomisation moments.

    Formula:

        ``G = sum_i sum_{j != i} w_ij x_i x_j / sum_i sum_{j != i} x_i x_j``

    with ``E[G] = S0 / (n(n-1))`` and the randomisation variance of
    Getis and Ord (1992),

        ``var(G) = [B0 m2^2 + B1 m4 + B2 m1^2 m2 + B3 m1 m3 + B4 m1^4]``
        ``        / [(m1^2 - m2)^2 n(n-1)(n-2)(n-3)] - E[G]^2``

    where ``m_r = sum_i x_i^r`` and, with ``S0 = sum_ij w_ij``,
    ``S1 = (1/2) sum_ij (w_ij + w_ji)^2``,
    ``S2 = sum_i (sum_j w_ij + sum_j w_ji)^2``,

        ``B0 = (n^2 - 3n + 3) S1 - n S2 + 3 S0^2``
        ``B1 = -[(n^2 - n) S1 - 2n S2 + 6 S0^2]``
        ``B2 = -[2n S1 - (n+3) S2 + 6 S0^2]``
        ``B3 = 4(n-1) S1 - 2(n+1) S2 + 8 S0^2``
        ``B4 = S1 - S2 + S0^2``.

    G is only interpretable for non-negative ``x`` -- the denominator is
    a sum of cross-products, so a mixed-sign variable can drive it
    through zero -- and negative input is rejected rather than silently
    returned as a large number.  Binary weights are what the statistic
    was designed for; a row-standardised ``W`` makes ``G`` a weighted
    mean rather than a hot-spot measure.

    Parameters
    ----------
    x : array-like
        Non-negative variable of length ``n``.
    W : array-like
        ``n x n`` spatial weights.  The diagonal is forced to zero.

    Returns
    -------
    RichResult
        ``estimate`` (G), ``statistic`` (standard deviate), ``p_value``
        (two-sided), ``expected``, ``var``, ``S0``, ``S1``, ``S2``,
        ``n``, ``method``.

    References
    ----------
    Getis and Ord (1992), The analysis of spatial association by use of
    distance statistics, Geographical Analysis 24:189-206.  Paywalled;
    the coded form of the moments was read from Bivand and Ono's
    ``spdep``, R/globalG.R and the ``spweights.constants`` helper in
    R/utils.R (tarball spdep_1.4-2 fetched from CRAN), which is the
    reference implementation.  ``B1`` is taken with spdep's
    ``B1correct = TRUE`` default, i.e. ``6 S0^2`` rather than the
    ``3 S0^2`` that CrimeStat IV uses.
    """
    x = T.vec(x)
    W = T.mat(W)
    n = len(x)
    if len(W) != n or any(len(row) != n for row in W):
        raise ValueError("W must be n x n with n = len(x)")
    if any(xi < 0 for xi in x):
        raise ValueError("Getis-Ord G is undefined for negative x")
    for i in range(n):
        W[i][i] = 0.0
    numer = sum(W[i][j] * x[i] * x[j] for i in range(n) for j in range(n))
    denom = sum(x[i] * x[j] for i in range(n) for j in range(n) if i != j)
    if denom == 0:
        raise ValueError("degenerate x: cross-product sum is zero")
    g = numer / denom
    s0 = sum(W[i][j] for i in range(n) for j in range(n))
    s1 = 0.0
    for i in range(n):
        for j in range(n):
            s1 += W[i][j] * W[i][j] + W[i][j] * W[j][i]
    rs = [sum(W[i][j] for j in range(n)) for i in range(n)]
    cs = [sum(W[i][j] for i in range(n)) for j in range(n)]
    s2 = sum((rs[i] + cs[i]) ** 2 for i in range(n))
    nn = float(n * n)
    n1, n2, n3 = n - 1.0, n - 2.0, n - 3.0
    eg = s0 / (n * n1)
    s02 = s0 * s0
    b0 = (nn - 3.0 * n + 3.0) * s1 - n * s2 + 3.0 * s02
    b1 = -((nn - n) * s1 - 2.0 * n * s2 + 6.0 * s02)
    b2 = -(2.0 * n * s1 - (n + 3.0) * s2 + 6.0 * s02)
    b3 = 4.0 * n1 * s1 - 2.0 * (n + 1.0) * s2 + 8.0 * s02
    b4 = s1 - s2 + s02
    m1 = sum(x)
    m2 = sum(xi ** 2 for xi in x)
    m3 = sum(xi ** 3 for xi in x)
    m4 = sum(xi ** 4 for xi in x)
    vg = ((b0 * m2 * m2 + b1 * m4 + b2 * m1 * m1 * m2 + b3 * m1 * m3 + b4 * m1 ** 4)
          / (((m1 * m1 - m2) ** 2) * n * n1 * n2 * n3)) - eg * eg
    z = (g - eg) / math.sqrt(vg) if vg > 0 else float("nan")
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if vg > 0 else float("nan")
    return RichResult(
        payload={
            "estimate": float(g),
            "statistic": float(z),
            "p_value": float(p),
            "expected": float(eg),
            "var": float(vg),
            "S0": float(s0),
            "S1": float(s1),
            "S2": float(s2),
            "n": int(n),
            "method": "Getis-Ord global G",
        }
    )


def cheatsheet():
    return "getis_ord_g(x, W): global G with the Getis-Ord randomisation moments."


# No compact alias here: ``getis_ord_g`` and ``getisordg`` are already
# registered in _lazy_map.json against the duplicate module ``getisg``,
# so a second registration would shadow one of the two.  See the batch
# report -- the duplication is a defect, not a design.
