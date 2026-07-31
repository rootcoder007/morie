"""Relationship between covariance and semivariogram under stationarity."""

import numpy as np

from ._richresult import RichResult
from ._schab_vario import _as_lag

__all__ = ["schabenberger_stationary_cov_semivario"]


def schabenberger_stationary_cov_semivario(cov_func, h):
    r"""
    Covariance and semivariogram of a second-order stationary field.

    For a second-order stationary process,

    .. math::

        \gamma(h) = C(0) - C(h)

    so the semivariogram rises to the sill :math:`C(0)` exactly as the
    covariance decays. The identity requires second-order stationarity:
    an intrinsically stationary process has a semivariogram but need not
    have a covariance function at all.

    Parameters
    ----------
    cov_func : callable
        ``C(h)``, accepting an array of lags and returning an array.
    h : array-like
        Lag distances, non-negative.

    Returns
    -------
    RichResult
        ``gamma``, ``covariance``, ``sill`` (:math:`C(0)`).

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 1.4.2 / Ch. 2.
    """
    if not callable(cov_func):
        raise TypeError("`cov_func` must be callable, C(h) -> array")
    h = _as_lag(h)
    c0 = float(np.asarray(cov_func(np.zeros(1))).ravel()[0])
    ch = np.asarray(cov_func(h), dtype=float).ravel()
    return RichResult(
        title="Covariance and semivariogram under second-order stationarity",
        summary_lines=[("sill C(0)", c0)],
        payload={"gamma": c0 - ch, "covariance": ch, "sill": c0},
    )


def cheatsheet():
    return "spssoc: gamma(h) = C(0) - C(h) for a second-order stationary field."
