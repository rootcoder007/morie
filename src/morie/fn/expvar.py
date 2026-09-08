"""Exponential semivariogram model."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["exponential_variogram_model"]


def exponential_variogram_model(h, c0=0.0, c=1.0, a=1.0):
    r"""Exponential semivariogram, in the range-parameter parameterisation.

    .. math::
        \gamma(h) = c_0 + c\left(1 - e^{-h/a}\right), \quad h > 0,
        \qquad \gamma(0) = 0.

    :math:`c_0` is the nugget, :math:`c` the partial sill (so the sill is
    :math:`c_0 + c`), and :math:`a` the range PARAMETER. The nugget is a
    discontinuity at the origin: :math:`\gamma(0) = 0` by definition even
    when :math:`c_0 > 0`, and :math:`\gamma(0^+) = c_0`.

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
        ``gamma``, ``h``, ``c0``, ``c``, ``a``, ``sill``,
        ``practical_range``, ``covariance`` (:math:`C(h)`), ``n``,
        ``method``.

    Notes
    -----
    Beware two live conventions for the range. This module uses the range
    PARAMETER ``a`` of the formula above, which is what the module's own
    specification asks for. Schabenberger & Gotway write the same model
    on the PRACTICAL range :math:`\alpha`, eq. (4.11) p. 144:
    :math:`C(h) = \sigma^2\exp\{-\theta h\} = \sigma^2\exp\{-3h/\alpha\}`,
    so :math:`a = 1/\theta = \alpha/3` and the correlation has fallen to
    :math:`e^{-3} = 0.0498` at :math:`h = \alpha`. The shared core
    ``_schab_vario`` uses :math:`\alpha`; pass ``a = alpha / 3`` to move
    between them. Mixing the two silently rescales every fitted range by
    a factor of three.

    References
    ----------
    Cressie, N. A. C. (1993). *Statistics for Spatial Data*, rev. edn.
    Wiley, sec. 2.3.1, the exponential variogram model.

    Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
    Spatial Data Analysis*. Chapman & Hall/CRC, eq. (4.11), p. 144, which
    states both parameterisations and identifies alpha as the practical
    range. (Cressie 1993 is not in the local corpus; the parameterisation
    was verified against this rendered page.)
    """
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

    gamma = [0.0 if v == 0.0 else c0 + c * (1.0 - math.exp(-v / a)) for v in hs]
    cov = [c0 + c if v == 0.0 else c * math.exp(-v / a) for v in hs]

    return RichResult(
        payload={
            "gamma": gamma,
            "covariance": cov,
            "h": hs,
            "c0": c0,
            "c": c,
            "a": a,
            "sill": c0 + c,
            "practical_range": 3.0 * a,
            "n": len(hs),
            "method": "Exponential semivariogram, gamma(h) = c0 + c(1 - exp(-h/a))",
        }
    )


def _lags(h):
    if hasattr(h, "tolist"):
        h = h.tolist()
    if isinstance(h, (int, float)):
        h = [h]
    hs = [float(v) for v in h]
    if not hs:
        raise ValueError("h is empty.")
    for v in hs:
        if not (v >= 0.0):
            raise ValueError("lag distances must be non-negative; got %r" % (v,))
    return hs


def cheatsheet():
    return "expvar: gamma(h) = c0 + c(1 - exp(-h/a)); practical range 3a."
