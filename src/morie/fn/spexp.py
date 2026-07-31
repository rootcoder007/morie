"""Exponential semivariogram model."""

import numpy as np

from ._richresult import RichResult
from ._schab_vario import semivariogram

__all__ = ["schabenberger_exponential_variogram"]


def schabenberger_exponential_variogram(h, nugget=0.0, sill=1.0, range=1.0):
    r"""
    Exponential semivariogram model.

    .. math::

        \gamma(h) = c_0 + \sigma_0^2\left(1 - e^{-3h/\alpha}\right),
        \qquad h > 0

    ``range`` is the PRACTICAL range in the book's parameterisation: the
    lag at which the correlation has fallen to :math:`e^{-3} = 0.049787`
    ("0.05 or less", p. 143).

    Parameters
    ----------
    h : array-like
        Lag distances, non-negative.
    nugget : float, default 0.0
        Nugget effect :math:`c_0`. A discontinuity AT the origin, so
        ``gamma(0) == 0`` even when ``nugget > 0``.
    sill : float, default 1.0
        Partial sill :math:`\sigma_0^2`. The total sill is
        ``nugget + sill``.
    range : float, default 1.0
        Practical range :math:`\alpha`, must be positive.

    Returns
    -------
    RichResult
        ``gamma`` (array), plus the echoed ``nugget``, ``sill``, ``range``
        and ``model``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Eq. (4.11), p. 144.
    """
    g = semivariogram(h, nugget, sill, range, "exponential")
    return RichResult(
        title="Exponential semivariogram model",
        summary_lines=[("nugget", nugget), ("partial sill", sill),
                       ("practical range", range)],
        payload={"gamma": g, "nugget": float(nugget), "sill": float(sill),
                 "range": float(range), "model": "exponential"},
    )


def cheatsheet():
    return "spexp: Exponential semivariogram model"
