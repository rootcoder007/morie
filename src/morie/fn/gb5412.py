# morie.fn -- function file (rootcoder007/morie)
"""Normal approximation with continuity correction for the sign test."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['signz', 'gibbons_sign_normal_approx']


def signz(k, n, alternative="two-sided", correct=True):
    """Standardised sign statistic, eq. (5.4.7).

    Book p. 174:

    .. math:: Z = \\frac{K - N/2 - 0.5}{\\sqrt{N/4}}
                = \\frac{2K - N - 1}{\\sqrt{N}},

    the continuity-corrected form used for the upper tail; the
    correction is applied toward the null in whichever direction K
    lies.  Set ``correct=False`` for the uncorrected (2K - N)/sqrt(N).

    Parameters
    ----------
    k, n : int
        Positive-difference count and number of non-zero differences.
    alternative : str, optional
        ``"two-sided"``, ``"greater"`` or ``"less"``.
    correct : bool, optional
        Apply the 0.5 continuity correction (default True).

    Returns
    -------
    RichResult
        keys ``z``, ``p_value``, ``statistic``, ``n``, ``mean``,
        ``var``, ``alternative``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (5.4.7), p. 174.
    """
    k = int(k)
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1.")
    if not 0 <= k <= n:
        raise ValueError("k must lie in 0..n.")
    mean = n / 2.0
    sd = math.sqrt(n / 4.0)
    d = k - mean
    if correct:
        if d > 0.0:
            d -= 0.5
        elif d < 0.0:
            d += 0.5
    z = d / sd
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
            "z": float(z),
            "p_value": float(min(1.0, pv)),
            "statistic": k,
            "n": n,
            "mean": mean,
            "var": n / 4.0,
            "alternative": alternative,
            "method": "sign test normal approximation, eq. (5.4.7)",
        }
    )


gibbons_sign_normal_approx = signz
