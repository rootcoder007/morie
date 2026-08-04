# morie.fn -- function file (rootcoder007/morie)
"""Power of the Wilcoxon signed-rank test by normal approximation."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['wsrpow', 'gibbons_wsrt_power']


def wsrpow(n, p1, p2, alpha=0.05):
    """Normal-approximation power of the upper-tailed signed-rank test.

    Book p. 205, eqs. (5.7.13)-(5.7.14).  With p1 = P(X_i > M0) and
    p2 = P(X_i + X_j > 2M0) under H1, the alternative mean of T+ is
    N p1 + N(N-1) p2 / 2, and taking Noether's r = sigma/sigma_0 = 1,

    .. math:: z_\\beta = \\frac{N(p_1-0.5) + N(N-1)(p_2-0.5)/2}
        {\\sqrt{N(N+1)(2N+1)/24}} - z_\\alpha,

    with power = Phi(z_beta).

    Parameters
    ----------
    n : int
        Sample size.
    p1 : float
        P(X_i > M0) under the alternative.
    p2 : float
        P(X_i + X_j > 2 M0) under the alternative, i < j.
    alpha : float, optional
        One-sided size (default 0.05).

    Returns
    -------
    RichResult
        keys ``power``, ``z_beta``, ``shift`` (mu - mu0), ``sd0``,
        ``n``, ``p1``, ``p2``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eqs. (5.7.13)-(5.7.14), p. 205
    (Noether, 1987).
    """
    n = int(n)
    p1 = float(p1)
    p2 = float(p2)
    alpha = float(alpha)
    if n < 2:
        raise ValueError("n must be at least 2.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1).")
    shift = n * (p1 - 0.5) + n * (n - 1.0) * (p2 - 0.5) / 2.0
    sd0 = math.sqrt(n * (n + 1.0) * (2.0 * n + 1.0) / 24.0)
    za = stats.norm.ppf(1.0 - alpha)
    zb = shift / sd0 - za
    return RichResult(
        payload={
            "power": float(stats.norm.cdf(zb)),
            "z_beta": float(zb),
            "shift": float(shift),
            "sd0": float(sd0),
            "n": n,
            "p1": p1,
            "p2": p2,
            "method": "signed-rank power, eqs. (5.7.13)-(5.7.14)",
        }
    )


gibbons_wsrt_power = wsrpow
