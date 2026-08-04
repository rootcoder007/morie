# morie.fn -- function file (rootcoder007/morie)
"""Normal approximation for the Wilcoxon rank-sum test."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['wrsz', 'gibbons_wrs_normal_approx']


def wrsz(w, m, n, alternative="two-sided", correct=False, ties=None):
    """Standardised rank-sum statistic, with the optional tie correction.

    Section 8.2 (book p. 291): E[W_N] = m(N+1)/2 and
    Var[W_N] = mn(N+1)/12.  When midranks are used the variance is
    reduced by the usual tie term,

    .. math:: Var[W_N] = \\frac{mn}{12}\\left[(N+1)
        - \\frac{\\sum t(t^2-1)}{N(N-1)}\\right],

    which is the two-sample analogue of eq. (5.7.11).

    Parameters
    ----------
    w : float
        Observed W_N.
    m, n : int
        The two sample sizes.
    alternative : str, optional
        ``"two-sided"``, ``"greater"`` or ``"less"``.
    correct : bool, optional
        Apply a 0.5 continuity correction (default False).
    ties : sequence of int, optional
        Multiplicities of the tied groups.

    Returns
    -------
    RichResult
        keys ``z``, ``p_value``, ``mean``, ``var``,
        ``var_uncorrected``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 8.2, p. 291; tie correction
    Sec. 8.2 following eq. (5.7.11), p. 203.
    """
    m = int(m)
    n = int(n)
    w = float(w)
    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")
    nn = m + n
    mean = m * (nn + 1.0) / 2.0
    v0 = m * n * (nn + 1.0) / 12.0
    var = v0
    if ties:
        s = sum(float(t) * (float(t) ** 2 - 1.0) for t in ties)
        var = m * n / 12.0 * ((nn + 1.0) - s / (nn * (nn - 1.0)))
    d = w - mean
    if correct:
        d = d - 0.5 if d > 0 else (d + 0.5 if d < 0 else d)
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
            "var_uncorrected": float(v0),
            "m": m,
            "n": n,
            "method": "rank-sum normal approximation (Sec. 8.2)",
        }
    )


gibbons_wrs_normal_approx = wrsz
