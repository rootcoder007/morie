# morie.fn -- function file (rootcoder007/morie)
"""Efficacy of a test statistic -- Gibbons eq. (13.2.4)."""

import math

from ._richresult import RichResult

__all__ = ['efficacy', 'gibbons_efficacy']


def efficacy(deriv, var):
    """e(T_n) = [dE(T_n)/dtheta]^2 / sigma^2(T_n).

    Book p. 486, eq. (13.2.4).  The efficacy measures how fast the mean
    of the statistic moves away from its null value per unit of null
    standard deviation squared; the ARE of Theorem 13.2.2 is the ratio
    of two efficacies, so all the ARE results of Ch. 13 are assembled
    from this one quantity.

    Parameters
    ----------
    deriv : float
        dE(T_n)/dtheta at theta = theta_0.
    var : float
        sigma^2(T_n) at theta = theta_0, strictly positive.

    Returns
    -------
    RichResult
        keys ``efficacy``, ``deriv``, ``var``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (13.2.4), p. 486.
    """
    d = float(deriv)
    v = float(var)
    if v <= 0.0:
        raise ValueError("var must be strictly positive.")
    return RichResult(
        payload={
            "efficacy": float(d * d / v),
            "deriv": d,
            "var": v,
            "method": "efficacy e(T) = [dE/dtheta]^2 / var, eq. (13.2.4)",
        }
    )


gibbons_efficacy = efficacy
