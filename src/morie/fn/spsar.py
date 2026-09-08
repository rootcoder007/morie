"""Spatial autoregressive (lag) model (delegates to spatial_ar_lag)."""

from .sarla import spatial_ar_lag

__all__ = ["schabenberger_sar_model"]


def schabenberger_sar_model(x, y, w):
    """
    Spatial autoregressive (lag) model.

    Same estimator as :func:`morie.fn.sarla.spatial_ar_lag`, which fits
    the concentrated log-likelihood in rho; this delegates rather than
    carrying a second implementation.

    Parameters
    ----------
    x : array-like
        Covariates, shape (n, p).
    y : array-like
        Response, shape (n,).
    w : array-like
        Spatial weights, shape (n, n).

    Returns
    -------
    The result of ``spatial_ar_lag``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 6.2.2.1
    "Simultaneous Autoregressive (SAR) Models", pp. 335-341.
    """
    return spatial_ar_lag(x, y, w)


def cheatsheet():
    return "spsar: SAR lag model; delegates to spatial_ar_lag (sarla)."
