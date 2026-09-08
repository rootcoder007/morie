"""Cokriging: multivariate prediction (delegates to cokriging)."""

from .cokrg import cokriging

__all__ = ["schabenberger_cokriging"]


def schabenberger_cokriging(coords, z1, z2, target, cross_cov_model=None):
    """
    Cokriging: multivariate prediction using primary and secondary variables.

    This is the same estimator as :func:`morie.fn.cokrg.cokriging` and
    delegates to it rather than carrying a second implementation.

    Parameters
    ----------
    coords : array-like
        Observation coordinates, shape (n, 2).
    z1, z2 : array-like
        Primary and secondary variable, each shape (n,).
    target : array-like
        Prediction location(s).
    cross_cov_model : mapping, optional
        Linear-model-of-coregionalization parameters. Recognised keys:
        ``sill_p``, ``range_p``, ``sill_s``, ``range_s``, ``cross_sill``,
        ``cross_range``, ``nugget``.

    Returns
    -------
    The result of ``cokriging``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Ch. 5.
    """
    keys = ("sill_p", "range_p", "sill_s", "range_s",
            "cross_sill", "cross_range", "nugget")
    cm = dict(cross_cov_model or {})
    kw = {k: cm[k] for k in keys if k in cm}
    return cokriging(z1, z2, coords, target, **kw)


def cheatsheet():
    return "spcokr: cokriging; delegates to cokriging (cokrg)."
