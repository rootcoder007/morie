# morie.fn -- function file (rootcoder007/morie)
"""Kendall's tau used as a test against trend in a time series."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['tautrend', 'gibbons_kendall_trend']


def tautrend(y, alternative="two-sided"):
    """Trend test from the concordance of Y with its time index.

    Section 11.2.5 (book p. 406).  A hypothesis of randomness in one
    time-ordered sequence is the same as independence between the
    sequence and the numbers 1, 2, ..., n, so with x_i = i the
    indicators of eq. (11.2.3) become

    .. math:: A_{ij} = \\mathrm{sgn}(j-i)\\,
        \\mathrm{sgn}(Y_j - Y_i),

    and tau over these pairs measures trend.  Unlike runs up and down,
    tau compares every observation with every earlier one, not only
    the immediately preceding one.

    Parameters
    ----------
    y : sequence of float
        Time-ordered observations, n >= 3.
    alternative : str, optional
        ``"two-sided"``, ``"greater"`` (upward trend) or ``"less"``.

    Returns
    -------
    RichResult
        keys ``tau``, ``statistic`` (S = P - Q), ``P``, ``Q``, ``z``,
        ``p_value``, ``var``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 11.2.5, p. 406, with
    eq. (11.2.3), p. 391.
    """
    ys = [float(v) for v in y]
    n = len(ys)
    if n < 3:
        raise ValueError("need at least 3 observations.")
    p = q = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = ys[j] - ys[i]
            if d > 0:
                p += 1
            elif d < 0:
                q += 1
    npairs = n * (n - 1) // 2
    s = p - q
    tau = s / float(npairs)
    var = 2.0 * (2.0 * n + 5.0) / (9.0 * n * (n - 1.0))
    z = tau / math.sqrt(var)
    if alternative == "greater":
        pv = 1.0 - stats.norm.cdf(z)
    elif alternative == "less":
        pv = stats.norm.cdf(z)
    elif alternative == "two-sided":
        pv = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    else:
        raise ValueError("alternative must be two-sided, greater or less.")
    return RichResult(
        payload={
            "tau": float(tau),
            "statistic": int(s),
            "P": int(p),
            "Q": int(q),
            "z": float(z),
            "p_value": float(min(1.0, pv)),
            "var": float(var),
            "n": n,
            "method": "Kendall tau trend test (Sec. 11.2.5)",
        }
    )


gibbons_kendall_trend = tautrend
