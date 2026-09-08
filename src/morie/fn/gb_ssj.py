# morie.fn -- function file (rootcoder007/morie)
"""Two-sided sample size for the sign test (alpha replaced by alpha/2)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['signnasy', 'gibbons_sign_sample_size_2']


def signnasy(theta, alpha=0.05, beta=0.10):
    """Sample size for a two-sided sign test.

    Book p. 179: "A sample size formula for the two-sided alternative
    is the same as (5.4.9) with alpha replaced by alpha/2."  So

    .. math:: N = \\left[\\frac{\\sqrt{\\theta(1-\\theta)}\\,z_\\beta
        + 0.5 z_{\\alpha/2}}{0.5 - \\theta}\\right]^{2}.

    Parameters
    ----------
    theta : float
        P(X > M0) under the alternative, theta != 0.5.
    alpha : float, optional
        Two-sided size (default 0.05).
    beta : float, optional
        Type II error (default 0.10).

    Returns
    -------
    RichResult
        keys ``n``, ``n_raw``, ``root_n``, ``z_alpha``, ``z_beta``,
        ``theta``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (5.4.9) two-sided form, p. 179.
    """
    theta = float(theta)
    alpha = float(alpha)
    beta = float(beta)
    if theta == 0.5:
        raise ValueError("theta must differ from 0.5.")
    if not 0.0 < theta < 1.0:
        raise ValueError("theta must lie strictly inside (0, 1).")
    za = stats.norm.ppf(1.0 - alpha / 2.0)
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
            "method": "two-sided sign test sample size, eq. (5.4.9) with alpha/2",
        }
    )


gibbons_sign_sample_size_2 = signnasy
