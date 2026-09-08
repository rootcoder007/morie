# morie.fn -- function file (rootcoder007/morie)
"""Efficacy of the two-sample Student t test -- Gibbons eq. (13.3.9)."""

import math

from ._richresult import RichResult

__all__ = ['efft2', 'gibbons_two_sample_t_efficacy']


def efft2(m, n, sigma2):
    """e(T*_{m,n}) = mn / [sigma^2 (m+n)].

    Book p. 494, eq. (13.3.9).  The pooled-variance two-sample t
    statistic has efficacy depending on the parent only through the
    common variance, for any continuous population.  Dividing the
    rank-sum efficacy of eq. (13.3.10) by this gives the classical
    ARE result 12 sigma^2 [int f^2]^2, returned as ``are_wrs`` for a
    unit integral so the caller can scale it.

    Parameters
    ----------
    m, n : int
        The two sample sizes.
    sigma2 : float
        Common population variance, strictly positive.

    Returns
    -------
    RichResult
        keys ``efficacy``, ``m``, ``n``, ``sigma2``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (13.3.9), p. 494.
    """
    m = int(m)
    n = int(n)
    s2 = float(sigma2)
    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")
    if s2 <= 0.0:
        raise ValueError("sigma2 must be strictly positive.")
    return RichResult(
        payload={
            "efficacy": float(m * n / (s2 * (m + n))),
            "m": m,
            "n": n,
            "sigma2": s2,
            "method": "two-sample t efficacy, eq. (13.3.9)",
        }
    )


gibbons_two_sample_t_efficacy = efft2
