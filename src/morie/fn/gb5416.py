# morie.fn -- function file (rootcoder007/morie)
"""Sample size for the sign test -- Gibbons eq. (5.4.9)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['signn', 'gibbons_sign_sampsize']


def signn(theta, alpha=0.05, beta=0.10):
    """Normal-approximation sample size for a one-sided sign test.

    Book p. 179, eq. (5.4.9):

    .. math:: N = \\left[\\frac{\\sqrt{\\theta(1-\\theta)}\\,z_\\beta
        + 0.5 z_\\alpha}{0.5 - \\theta}\\right]^{2},

    rounded up to the next integer.  The book's worked example takes
    theta = 0.2, alpha = 0.05, 1 - beta = 0.90 and gets sqrt(N) = 4.45,
    N = 19.8, hence 20 observations.

    Parameters
    ----------
    theta : float
        P(X > M0) under the alternative, theta != 0.5.
    alpha : float, optional
        Size (default 0.05).
    beta : float, optional
        Type II error, so power is 1 - beta (default 0.10).

    Returns
    -------
    RichResult
        keys ``n`` (rounded up), ``n_raw``, ``root_n``, ``z_alpha``,
        ``z_beta``, ``theta``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (5.4.9), p. 179.
    """
    theta = float(theta)
    alpha = float(alpha)
    beta = float(beta)
    if theta == 0.5:
        raise ValueError("theta must differ from 0.5.")
    if not 0.0 < theta < 1.0:
        raise ValueError("theta must lie strictly inside (0, 1).")
    za = stats.norm.ppf(1.0 - alpha)
    zb = stats.norm.ppf(1.0 - beta)
    root = (math.sqrt(theta * (1.0 - theta)) * zb + 0.5 * za) / (0.5 - theta)
    nraw = root * root
    return RichResult(
        payload={
            "n": int(math.ceil(nraw)),
            "n_raw": float(nraw),
            "root_n": float(abs(root)),
            "z_alpha": float(za),
            "z_beta": float(zb),
            "theta": theta,
            "method": "sign test sample size, eq. (5.4.9)",
        }
    )


gibbons_sign_sampsize = signn
