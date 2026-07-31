"""Power semivariogram model (unbounded)."""

import numpy as np

from ._richresult import RichResult
from ._schab_vario import _as_lag

__all__ = ["schabenberger_power_variogram"]


def schabenberger_power_variogram(h, nugget=0.0, c1=1.0, alpha=1.0):
    r"""
    Power semivariogram model (unbounded).

    .. math::

        \gamma(h) = \theta h^{\lambda}, \qquad \theta \ge 0,\; 0 \le \lambda < 2

    This model is NOT second-order stationary -- it has no sill. For
    :math:`\lambda = 1` it reduces to the linear semivariogram. The book is
    explicit that :math:`\lambda \ge 2` "violates the intrinsic
    hypothesis", so that is an error here rather than a silent result.

    Parameters
    ----------
    h : array-like
        Lag distances, non-negative.
    nugget : float, default 0.0
        Nugget effect, added for ``h > 0``.
    c1 : float, default 1.0
        Scale :math:`\theta`, must be non-negative.
    alpha : float, default 1.0
        Exponent :math:`\lambda`, in ``[0, 2)``.

    Returns
    -------
    RichResult
        ``gamma`` (array), plus ``nugget``, ``theta``, ``lambda``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 4.3.5, eq. (4.21),
    p. 149.
    """
    if c1 < 0:
        raise ValueError("`c1` (theta) must be >= 0")
    if not (0.0 <= alpha < 2.0):
        raise ValueError(
            "`alpha` (lambda) must satisfy 0 <= lambda < 2; lambda >= 2 "
            "violates the intrinsic hypothesis (Schabenberger & Gotway "
            "2005, p. 149)"
        )
    if nugget < 0:
        raise ValueError("`nugget` must be >= 0")
    h = _as_lag(h)
    g = nugget + c1 * np.power(h, alpha)
    g[h == 0] = 0.0
    return RichResult(
        title="Power semivariogram (unbounded)",
        summary_lines=[("nugget", nugget), ("theta", c1), ("lambda", alpha)],
        payload={"gamma": g, "nugget": float(nugget), "theta": float(c1),
                 "lambda": float(alpha), "model": "power"},
    )


def cheatsheet():
    return "sppow: power semivariogram, gamma(h) = theta h^lambda, 0 <= lambda < 2."
