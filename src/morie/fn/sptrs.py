"""Polynomial trend surface model (delegates to spatial_trend_surface)."""

from .sptrn import spatial_trend_surface

__all__ = ["schabenberger_trend_surface"]


def schabenberger_trend_surface(coords, z, poly_degree=2):
    """
    Polynomial trend surface model for a spatially varying mean.

    This is the same estimator as
    :func:`morie.fn.sptrn.spatial_trend_surface` and delegates to it rather
    than carrying a second implementation.

    Parameters
    ----------
    coords : array-like
        Coordinates, shape (n, 2).
    z : array-like
        Response, shape (n,).
    poly_degree : int, default 2
        Polynomial order of the trend surface.

    Returns
    -------
    The result of ``spatial_trend_surface``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 5.3.1 "Trend Surface
    Models".
    """
    return spatial_trend_surface(z, coords, order=int(poly_degree))


def cheatsheet():
    return ("sptrs: polynomial trend surface; delegates to "
            "spatial_trend_surface (sptrn).")
