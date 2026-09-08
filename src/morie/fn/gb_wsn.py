# morie.fn -- function file (rootcoder007/morie)
"""Normal approximation for the signed-rank test -- eq. (5.7.9)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['wsrz', 'gibbons_wsrt_normal_approx']


def wsrz(tplus, n, alternative="two-sided", correct=False):
    """Standardised signed-rank statistic.

    Book p. 202, eq. (5.7.9):

    .. math:: Z = \\frac{4T^+ - N(N+1)}{\\sqrt{2N(N+1)(2N+1)/3}},

    which is exactly (T+ - N(N+1)/4)/sqrt(N(N+1)(2N+1)/24).  A 0.5
    continuity correction, applied toward the null, is optional; the
    book notes it "generally improves the approximation".

    Parameters
    ----------
    tplus : float
        Observed T+.
    n : int
        Number of non-zero differences.
    alternative : str, optional
        ``"two-sided"``, ``"greater"`` or ``"less"``.
    correct : bool, optional
        Apply the continuity correction (default False).

    Returns
    -------
    RichResult
        keys ``z``, ``p_value``, ``mean``, ``var``, ``statistic``,
        ``n``, ``alternative``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (5.7.9), p. 202.
    """
    n = int(n)
    tplus = float(tplus)
    if n < 1:
        raise ValueError("n must be at least 1.")
    mean = n * (n + 1.0) / 4.0
    var = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0
    d = tplus - mean
    if correct:
        if d > 0.0:
            d -= 0.5
        elif d < 0.0:
            d += 0.5
    z = d / math.sqrt(var)
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
            "mean": float(mean),
            "var": float(var),
            "statistic": tplus,
            "n": n,
            "alternative": alternative,
            "method": "signed-rank normal approximation, eq. (5.7.9)",
        }
    )


gibbons_wsrt_normal_approx = wsrz
