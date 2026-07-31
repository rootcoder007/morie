"""Block kriging: prediction for areal unit B (delegates to spatial_block_kriging)."""

from .spblk import spatial_block_kriging

__all__ = ["schabenberger_block_kriging"]


def schabenberger_block_kriging(coords, z, blocks, cov_model=None):
    """
    Block kriging: prediction for areal unit B.

    This is the same estimator as :func:`morie.fn.spblk.spatial_block_kriging`
    and delegates to it rather than carrying a second implementation. The
    argument order here is the one this module has always exposed.

    Parameters
    ----------
    coords : array-like
        Observation coordinates, shape (n, 2).
    z : array-like
        Observed values at ``coords``, shape (n,).
    blocks : array-like
        Block (areal unit) definitions, as accepted by
        ``spatial_block_kriging``.
    cov_model : mapping, optional
        Covariance parameters. Recognised keys: ``nugget``, ``sill``,
        ``range_`` (or ``range``), ``n_quad``.

    Returns
    -------
    The result of ``spatial_block_kriging``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 5.7.1 "Block Kriging",
    building on the ordinary kriging system of Sec. 5.2.
    """
    kw = {}
    cm = dict(cov_model or {})
    if "range" in cm and "range_" not in cm:
        cm["range_"] = cm.pop("range")
    for k in ("nugget", "sill", "range_", "n_quad"):
        if k in cm:
            kw[k] = cm[k]
    return spatial_block_kriging(z, coords, blocks, **kw)


def cheatsheet():
    return ("spblkk: block kriging for an areal unit; delegates to "
            "spatial_block_kriging (spblk).")
