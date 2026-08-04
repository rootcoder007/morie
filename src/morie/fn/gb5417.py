# morie.fn -- function file (rootcoder007/morie)
"""Confidence interval for the median by inverting the sign test."""

import math

from ._richresult import RichResult

__all__ = ['signmedci', 'gibbons_sign_median_ci']


def signmedci(x, alpha=0.05):
    """Order-statistic confidence interval for the median.

    Book p. 179, eq. (5.4.11): the endpoints are X_(r) and X_(s) with
    s = N - r + 1, where r is the largest integer satisfying

    .. math:: \\sum_{i=0}^{r-1} \\binom{N}{i} (0.5)^N \\le \\alpha/2.

    The realised confidence coefficient is 1 - 2 times that tail, and
    is reported as ``coverage`` because the discreteness of the
    binomial makes it exceed 1 - alpha in general.

    Parameters
    ----------
    x : sequence of float
        Sample, n >= 2.
    alpha : float, optional
        Nominal two-sided level (default 0.05).

    Returns
    -------
    RichResult
        keys ``lower``, ``upper``, ``r``, ``s``, ``coverage``,
        ``tail``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (5.4.11), p. 179.
    """
    xs = sorted(float(v) for v in x)
    n = len(xs)
    alpha = float(alpha)
    if n < 2:
        raise ValueError("need at least 2 observations.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1).")
    half = 0.5**n
    r = 0
    tail = 0.0
    for cand in range(1, n + 1):
        t = sum(math.comb(n, i) for i in range(cand)) * half
        if t <= alpha / 2.0:
            r = cand
            tail = t
        else:
            break
    if r == 0:
        return RichResult(
            payload={
                "lower": float("nan"), "upper": float("nan"), "r": 0, "s": 0,
                "coverage": float("nan"), "tail": 0.0, "n": n,
                "method": "sign-test median CI: n too small for alpha",
            }
        )
    s = n - r + 1
    return RichResult(
        payload={
            "lower": xs[r - 1],
            "upper": xs[s - 1],
            "r": int(r),
            "s": int(s),
            "coverage": float(1.0 - 2.0 * tail),
            "tail": float(tail),
            "n": n,
            "method": "median CI from sign-test inversion, eq. (5.4.11)",
        }
    )


gibbons_sign_median_ci = signmedci
