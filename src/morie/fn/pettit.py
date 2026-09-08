# morie.fn -- function file (rootcoder007/morie)
"""Pettitt change-point test."""

from __future__ import annotations

import math

from . import _t4core as T

from ._richresult import RichResult

__all__ = ["pettitt_test"]


def pettitt_test(x):
    """Pettitt's non-parametric test for a single change point.

    Formula: with ``r_i`` the midranks of ``x``,

        ``U_k = 2 sum_{i<=k} r_i - k(n+1)``,  ``k = 1..n``

    the statistic is ``U* = max_k |U_k|``, the change point is the ``k``
    attaining it, and the two-sided p-value is approximated by

        ``p = min(1, 2 exp(-6 U*^2 / (n^3 + n^2)))``.

    The rank form is algebraically the Mann-Whitney form
    ``sum_{i<=k} sum_{j>k} sign(x_i - x_j)`` but is O(n) per split rather
    than O(n^2), so no separate double loop is carried here.  The
    approximation is only trustworthy for ``p <= 0.5``, which is why it
    is clamped rather than extrapolated.

    Parameters
    ----------
    x : array-like
        Series in time order.

    Returns
    -------
    RichResult
        ``statistic`` (U*), ``p_value``, ``changepoint`` (1-based),
        ``U``, ``n``, ``method``.

    References
    ----------
    Pettitt (1979), A non-parametric approach to the change point
    problem, JRSS C (Applied Statistics) 28:126-135.  Paywalled; the
    coded form was read from Pohlert's CRAN package ``trend``
    (R/pettitt.test.R, tarball trend_1.1.7 fetched from CRAN), which
    follows Verstraeten et al. (2006) in using the rank form and gives
    ``pval <- min(1, 2.0 * exp((-6.0 * U^2) / (n^3 + n^2)))``.
    """
    x = T.vec(x)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 observations")
    r = T.ranks(x)
    uk = []
    acc = 0.0
    for k in range(1, n + 1):
        acc += r[k - 1]
        uk.append(2.0 * acc - k * (n + 1.0))
    ustar = max(abs(v) for v in uk)
    kstar = min(k + 1 for k in range(n) if abs(uk[k]) == ustar)
    p = min(1.0, 2.0 * math.exp(-6.0 * ustar * ustar / (n ** 3 + n ** 2)))
    return RichResult(
        payload={
            "statistic": float(ustar),
            "p_value": float(p),
            "changepoint": int(kstar),
            "U": uk,
            "n": int(n),
            "method": "Pettitt single change-point test",
        }
    )


def cheatsheet():
    return "pettitt_test(x): U_k = 2 sum r_i - k(n+1); U* = max|U_k|."


# compact alias per ledger/NAMING.md
pettitttest = pettitt_test
