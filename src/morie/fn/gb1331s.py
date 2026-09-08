# morie.fn -- function file (rootcoder007/morie)
"""Efficacy of the sign test -- Gibbons eq. (13.3.3)."""

import math

from ._richresult import RichResult

__all__ = ['effsign', 'gibbons_sign_efficacy']


def effsign(n, fmed):
    """e(K_N) = 4 N f^2(theta) for the one-sample sign test.

    Book p. 489, eq. (13.3.3): for N observations from any continuous
    F_X with median theta,

    .. math:: e(K_N) = 4N f_X^2(\\theta)
        = 4N f^2[F^{-1}(0.5)].

    Only the density at the median enters, which is why the sign test
    can beat the t test for heavy-tailed parents.

    Parameters
    ----------
    n : int
        Sample size.
    fmed : float
        The parent density at the median, strictly positive.

    Returns
    -------
    RichResult
        keys ``efficacy``, ``per_obs`` (efficacy / N), ``n``,
        ``fmed``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (13.3.3), p. 489.
    """
    n = int(n)
    f = float(fmed)
    if n < 1:
        raise ValueError("n must be at least 1.")
    if f <= 0.0:
        raise ValueError("fmed must be strictly positive.")
    e = 4.0 * n * f * f
    return RichResult(
        payload={
            "efficacy": float(e),
            "per_obs": float(e / n),
            "n": n,
            "fmed": f,
            "method": "sign test efficacy, eq. (13.3.3)",
        }
    )


gibbons_sign_efficacy = effsign
