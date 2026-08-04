# morie.fn -- function file (rootcoder007/morie)
"""Exact binomial p-value for the sign test."""

import math

from ._richresult import RichResult

__all__ = ['signp', 'gibbons_sign_pvalue']


def signp(k, n, alternative="two-sided"):
    """Exact sign-test p-value from the Binomial(N, 1/2) null.

    Section 5.4 (book p. 169), eq. (5.4.3).  The two-sided p-value is
    the usual doubled smaller tail, capped at 1.

    Parameters
    ----------
    k : int
        Observed number of positive differences, 0 <= k <= n.
    n : int
        Number of non-zero differences.
    alternative : str, optional
        ``"two-sided"``, ``"greater"`` (H1: M > M0) or ``"less"``.

    Returns
    -------
    RichResult
        keys ``p_value``, ``p_lower``, ``p_upper``, ``statistic``,
        ``n``, ``alternative``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 5.4, eq. (5.4.3), p. 169.
    """
    k = int(k)
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1.")
    if not 0 <= k <= n:
        raise ValueError("k must lie in 0..n.")
    half = 0.5**n
    lower = sum(math.comb(n, i) for i in range(k + 1)) * half
    upper = sum(math.comb(n, i) for i in range(k, n + 1)) * half
    if alternative == "greater":
        pv = upper
    elif alternative == "less":
        pv = lower
    elif alternative == "two-sided":
        pv = min(1.0, 2.0 * min(lower, upper))
    else:
        raise ValueError("alternative must be two-sided, greater or less.")
    return RichResult(
        payload={
            "p_value": float(pv),
            "p_lower": float(lower),
            "p_upper": float(upper),
            "statistic": k,
            "n": n,
            "alternative": alternative,
            "method": "exact sign test, K ~ Bin(n, 1/2)",
        }
    )


gibbons_sign_pvalue = signp
