"""Geographically weighted regression (delegates to gwreg)."""

from .gwreg import geographically_weighted_regression

__all__ = ["schabenberger_gwr"]


def schabenberger_gwr(x, y, coords, bandwidth=None, kernel="gaussian"):
    """
    Geographically weighted regression: locally varying coefficients.

    Same estimator as
    :func:`morie.fn.gwreg.geographically_weighted_regression`; this
    delegates rather than carrying a second implementation.

    The book fits the model locally in the spatial sense while allowing
    general covariates -- the weights in W(s0) determine how far each
    observation influences the coefficients estimated at s0, and the
    locality is based on spatial position, not on covariate values.

    Parameters
    ----------
    x : array-like
        Covariates, shape (n, p).
    y : array-like
        Response, shape (n,).
    coords : array-like
        Coordinates, shape (n, 2).
    bandwidth : float, optional
        Kernel bandwidth.
    kernel : str, default 'gaussian'
        Kernel family.

    Returns
    -------
    The result of ``geographically_weighted_regression``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 6.1.3 "Spatially
    Explicit Models", eq. (6.9), pp. 316-317, citing Fotheringham et al.
    (2002).
    """
    return geographically_weighted_regression(x, y, coords, bandwidth, kernel)


def cheatsheet():
    return "spgwr: GWR; delegates to geographically_weighted_regression (gwreg)."
