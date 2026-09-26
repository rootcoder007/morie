"""Circular semivariogram model."""

from ._richresult import RichResult
from ._schab_vario import semivariogram

__all__ = ["schabenberger_circular_variogram"]


def schabenberger_circular_variogram(h, nugget=0.0, sill=1.0, range=1.0):
    r"""
    Circular semivariogram model.

    The circular correlation is the member of the spherical family that
    is valid in :math:`R^2` (Schabenberger & Gotway 2005, Sec. 4.3.3,
    p. 146):

    .. math::

        R_2(h) = \frac{2}{\pi}\left\{\arccos\frac{h}{\alpha}
        - \frac{h}{\alpha}\sqrt{1 - \frac{h^2}{\alpha^2}}\right\},
        \quad 0 \le h \le \alpha,

    and zero beyond the true range :math:`\alpha`, with
    :math:`\gamma(h) = c_0 + \sigma_0^2 \{1 - R_2(h)\}`. It is gstat's
    ``"Cir"`` model.

    Parameters
    ----------
    h : array-like
        Lag distances, non-negative.
    nugget : float, default 0.0
        Nugget effect :math:`c_0`, a discontinuity at the origin, so
        ``gamma(0) == 0`` even when ``nugget > 0``.
    sill : float, default 1.0
        Partial sill :math:`\sigma_0^2`.
    range : float, default 1.0
        True range :math:`\alpha`, where the correlation reaches zero.

    Returns
    -------
    RichResult
        ``gamma`` (array), plus the echoed ``nugget``, ``sill``, ``range``
        and ``model``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 4.3.3, p. 146.
    """
    g = semivariogram(h, nugget, sill, range, "circular")
    return RichResult(
        title="Circular semivariogram model",
        summary_lines=[("nugget", nugget), ("partial sill", sill), ("range", range)],
        payload={"gamma": g, "nugget": float(nugget), "sill": float(sill), "range": float(range), "model": "circular"},
    )


def cheatsheet():
    return "spcirc: Circular semivariogram model, valid in R^2"
