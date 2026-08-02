"""Nugget effect in the semivariogram: discontinuity at the origin."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_vario import semivariogram, _as_lag

__all__ = ["schabenberger_nugget_effect"]


def schabenberger_nugget_effect(h, nugget=0.0, sill=1.0, range=1.0,
                                model="exponential"):
    r"""
    Nugget effect in the semivariogram: discontinuity at the origin.

    Writing :math:`Z(s) = \sqrt{c_0}\,U_1(s) + \sigma_0 U_2(s)` with
    :math:`U_1` white noise of unit variance and :math:`U_2` second-order
    stationary with unit-sill semivariogram :math:`\gamma_2`,

    .. math::

        \gamma_z(h) = c_0 + \sigma_0^2 \gamma_2(h),
        \qquad \mathrm{Var}[Z(s)] = c_0 + \sigma_0^2

    The point of the model is the JUMP: :math:`\gamma(0) = 0` by
    definition, while :math:`\lim_{h \to 0^+} \gamma(h) = c_0`. This
    function returns both, so the discontinuity is explicit rather than
    implied.

    Parameters
    ----------
    h : array-like
        Lag distances, non-negative.
    nugget : float, default 0.0
        Nugget :math:`c_0`.
    sill : float, default 1.0
        Partial sill :math:`\sigma_0^2`.
    range : float, default 1.0
        Practical range of the nested continuous component.
    model : {'exponential', 'gaussian', 'spherical'}
        The unit-sill component :math:`\gamma_2`.

    Returns
    -------
    RichResult
        ``gamma``, ``gamma_at_zero`` (0.0), ``limit_at_zero_plus``
        (the nugget), ``total_sill``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 4.3.6, p. 150.
    """
    h = _as_lag(h)
    g = semivariogram(h, nugget, sill, range, model)
    return RichResult(
        title="Nugget effect (discontinuity at the origin)",
        summary_lines=[("gamma(0)", 0.0), ("limit as h -> 0+", float(nugget)),
                       ("total sill", float(nugget) + float(sill))],
        payload={"gamma": g, "gamma_at_zero": 0.0,
                 "limit_at_zero_plus": float(nugget),
                 "nugget": float(nugget), "sill": float(sill),
                 "total_sill": float(nugget) + float(sill), "model": model},
    )


def cheatsheet():
    return "spnug: nugget effect; gamma(0)=0 but gamma(0+)=c0."
