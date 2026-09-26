"""Tent (linear) semivariogram model."""

from ._richresult import RichResult
from ._schab_vario import semivariogram

__all__ = ["schabenberger_tent_variogram"]


def schabenberger_tent_variogram(h, nugget=0.0, sill=1.0, range=1.0):
    r"""
    Tent semivariogram model.

    The tent correlation is the member of the spherical family that is
    valid in :math:`R^1` (Schabenberger & Gotway 2005, Sec. 4.3.3, p. 146):

    .. math::

        R_1(h) = 1 - \frac{h}{\alpha},\quad 0 \le h \le \alpha,
        \qquad R_1(h) = 0,\quad h > \alpha,

    so :math:`\gamma(h) = c_0 + \sigma_0^2 h/\alpha` up to the true range
    :math:`\alpha` and the sill beyond it. It is gstat's ``"Lin"`` model
    with a range. It is a valid covariance only in one dimension.

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
    g = semivariogram(h, nugget, sill, range, "tent")
    return RichResult(
        title="Tent semivariogram model",
        summary_lines=[("nugget", nugget), ("partial sill", sill), ("range", range)],
        payload={"gamma": g, "nugget": float(nugget), "sill": float(sill), "range": float(range), "model": "tent"},
    )


def cheatsheet():
    return "sptent: Tent (linear) semivariogram model, valid in R^1"
