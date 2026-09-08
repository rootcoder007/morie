# morie.fn -- function file (rootcoder007/morie)
"""Confidence interval for the discordant proportion difference."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['mcnemarci', 'gibbons_mcnemar_ci']


def mcnemarci(table, alpha=0.05):
    """Interval for theta_12 - theta_21 from the McNemar variance.

    Book p. 523, eq. (14.5.2) and the derivation following it.  With
    T = (X_12 - X_21)/N, the book derives E[T] = theta_12 - theta_21
    and, from the multinomial covariance,

    .. math:: Var[T] = \\frac{(\\theta_{12}+\\theta_{21})
        - (\\theta_{12}-\\theta_{21})^2}{N},

    which reduces to (theta_12 + theta_21)/N under H0 -- the variance
    the chi-square form of eq. (14.5.1) uses.  Substituting the sample
    proportions gives the Wald interval returned here.

    Parameters
    ----------
    table : sequence of sequence of float
        The 2 x 2 table of paired counts.
    alpha : float, optional
        Two-sided level (default 0.05).

    Returns
    -------
    RichResult
        keys ``estimate``, ``lower``, ``upper``, ``se``,
        ``se_null`` (the H0 standard error), ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 14.5, eq. (14.5.2), p. 523.
    """
    tb = [[float(v) for v in row] for row in table]
    if len(tb) != 2 or any(len(row) != 2 for row in tb):
        raise ValueError("table must be 2 x 2.")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1).")
    nn = sum(sum(row) for row in tb)
    if nn <= 0:
        raise ValueError("the table must contain positive counts.")
    p12 = tb[0][1] / nn
    p21 = tb[1][0] / nn
    est = p12 - p21
    var = (p12 + p21 - est * est) / nn
    se = math.sqrt(max(0.0, var))
    sen = math.sqrt((p12 + p21) / nn)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return RichResult(
        payload={
            "estimate": float(est),
            "lower": float(est - z * se),
            "upper": float(est + z * se),
            "se": float(se),
            "se_null": float(sen),
            "n": float(nn),
            "method": "McNemar CI for theta_12 - theta_21 (Sec. 14.5)",
        }
    )


gibbons_mcnemar_ci = mcnemarci
