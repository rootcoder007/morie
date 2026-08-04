# morie.fn -- function file (rootcoder007/morie)
"""Efficacy of the one-sample Student t test -- Gibbons eq. (13.3.2)."""

import math

from ._richresult import RichResult

__all__ = ['efft', 'gibbons_t_efficacy']


def efft(n, sigma2):
    """e(T*_N) = N / sigma^2 for the one-sample t test.

    Book p. 488, eq. (13.3.2).  The classical test's efficacy depends
    on the parent only through its variance, which is exactly why it
    collapses relative to rank tests when the parent has heavy tails
    but a finite variance.  For paired data (book p. 493) sigma^2 is
    the variance of the differences,
    sigma_D^2 = sigma_X^2 + sigma_Y^2 - 2 Cov(X, Y).

    Parameters
    ----------
    n : int
        Sample size.
    sigma2 : float
        Population variance, strictly positive.

    Returns
    -------
    RichResult
        keys ``efficacy``, ``per_obs``, ``n``, ``sigma2``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (13.3.2), p. 488; paired case
    p. 493.
    """
    n = int(n)
    s2 = float(sigma2)
    if n < 1:
        raise ValueError("n must be at least 1.")
    if s2 <= 0.0:
        raise ValueError("sigma2 must be strictly positive.")
    return RichResult(
        payload={
            "efficacy": float(n / s2),
            "per_obs": float(1.0 / s2),
            "n": n,
            "sigma2": s2,
            "method": "one-sample t efficacy, eq. (13.3.2)",
        }
    )


gibbons_t_efficacy = efft
