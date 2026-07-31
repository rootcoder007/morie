"""Spatial Durbin model (delegates to spatial_durbin_model)."""

from .sgdbn import spatial_durbin_model

__all__ = ["schabenberger_spatial_durbin_model"]


def schabenberger_spatial_durbin_model(x, y, w):
    """
    Spatial Durbin model: SAR with spatially lagged covariates.

    This is the same estimator as
    :func:`morie.fn.sgdbn.spatial_durbin_model` and delegates to it rather
    than carrying a second implementation.

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
    The result of ``spatial_durbin_model``.

    References
    ----------
    LeSage, J. and Pace, R. K. (2009) Introduction to Spatial Econometrics.
    Chapman and Hall/CRC. doi:10.1201/9781420064254
    Bivand, R. S., Pebesma, E., and Gomez-Rubio, V. (2013) Applied Spatial
    Data Analysis with R, 2nd ed., Springer. Sec. 9.4.2 "Spatial
    Econometrics Approaches", pp. 307-311.
    NOT in Schabenberger & Gotway (2005): "Durbin" appears there only in
    the reference list.
    """
    return spatial_durbin_model(y, x, w)


def cheatsheet():
    return ("spsdm: spatial Durbin model; delegates to "
            "spatial_durbin_model (sgdbn).")
