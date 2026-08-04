"""Gaussian semivariogram model."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["gaussian_variogram_model"]


def gaussian_variogram_model(h, c0=0.0, c=1.0, a=1.0):
    r"""Gaussian semivariogram, in the range-parameter parameterisation.

    .. math::
        \gamma(h) = c_0 + c\left(1 - e^{-(h/a)^2}\right), \quad h > 0,
        \qquad \gamma(0) = 0.

    The gaussian model is the smoothest of the standard family -- it is
    infinitely differentiable at the origin, which is why Schabenberger &
    Gotway call the processes it describes "truly artificial" (p. 144)
    and warn that the name has nothing to do with the Gaussian
    distribution. It approaches its sill parabolically rather than
    linearly, so it fits short-lag behaviour very differently from the
    exponential model.

    The previous body was a placeholder: it averaged ``h`` and never used
    ``c0``, ``c`` or ``a``.

    Parameters
    ----------
    h : float or array-like
        Lag distances, all non-negative.
    c0 : float, default 0.0
        Nugget, >= 0.
    c : float, default 1.0
        Partial sill, >= 0.
    a : float, default 1.0
        Range parameter, > 0.

    Returns
    -------
    RichResult
        ``gamma``, ``covariance``, ``h``, ``c0``, ``c``, ``a``, ``sill``,
        ``practical_range``, ``n``, ``method``.

    Notes
    -----
    As for the exponential model there are two range conventions. Here
    ``a`` is the range PARAMETER of the formula above. Schabenberger &
    Gotway's eq. (4.10) uses the PRACTICAL range :math:`\alpha`, with
    :math:`R(h) = \exp\{-3(h/\alpha)^2\}`, so
    :math:`a = \alpha/\sqrt{3}` and the practical range is
    :math:`\alpha = a\sqrt{3}`. Note the exponential model's factor is
    :math:`\alpha/3`, not :math:`\alpha/\sqrt3` -- the two conversions
    differ, which is an easy way to misfit a range by 70%.

    References
    ----------
    Cressie, N. A. C. (1993). *Statistics for Spatial Data*, rev. edn.
    Wiley, sec. 2.3.1, the gaussian variogram model.

    Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
    Spatial Data Analysis*. Chapman & Hall/CRC, eq. (4.10) and the
    discussion of it on p. 144.
    """
    from .expvar import _lags

    hs = _lags(h)
    c0 = float(c0)
    c = float(c)
    a = float(a)
    if c0 < 0.0:
        raise ValueError("c0 (nugget) must be >= 0")
    if c < 0.0:
        raise ValueError("c (partial sill) must be >= 0")
    if not (a > 0.0):
        raise ValueError("a (range parameter) must be > 0")

    gamma = [0.0 if v == 0.0 else c0 + c * (1.0 - math.exp(-((v / a) ** 2)))
             for v in hs]
    cov = [c0 + c if v == 0.0 else c * math.exp(-((v / a) ** 2)) for v in hs]

    return RichResult(
        payload={
            "gamma": gamma,
            "covariance": cov,
            "h": hs,
            "c0": c0,
            "c": c,
            "a": a,
            "sill": c0 + c,
            "practical_range": math.sqrt(3.0) * a,
            "n": len(hs),
            "method": "Gaussian semivariogram, gamma(h) = c0 + c(1 - exp(-(h/a)^2))",
        }
    )


def cheatsheet():
    return "gauvar: gamma(h) = c0 + c(1 - exp(-(h/a)^2)); practical range a*sqrt(3)."
