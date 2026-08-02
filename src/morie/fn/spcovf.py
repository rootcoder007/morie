"""Spatial covariance function C(h) = Cov[Z(s), Z(s+h)]."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_vario import empirical_semivariogram

__all__ = ["schabenberger_covariance_function"]


def schabenberger_covariance_function(coords, z, n_bins=15, max_dist=None):
    r"""
    Empirical covariance function of a second-order stationary field.

    For a second-order stationary random field the covariance depends on
    the lag alone,

    .. math::  C(h) = \mathrm{Cov}[Z(s), Z(s+h)]

    with :math:`C(0) = \mathrm{Var}[Z(s)]` the sill. It is estimated here
    by binning pairs on lag and averaging the centred cross-products.

    The relationship to the semivariogram is reported alongside, because
    it only holds under SECOND-ORDER stationarity:

    .. math::  \gamma(h) = C(0) - C(h)

    An intrinsically stationary process has a semivariogram but need not
    have a covariance function at all, so the empirical semivariogram is
    computed independently rather than derived from :math:`C`, and the
    gap between the two is returned as a diagnostic.

    Parameters
    ----------
    coords : array-like
        Coordinates, shape ``(n, d)``.
    z : array-like
        Observed values, shape ``(n,)``.
    n_bins : int, default 15
        Number of lag bins.
    max_dist : float, optional
        Largest lag retained; half the maximum pair distance by default.

    Returns
    -------
    RichResult
        ``lag``, ``covariance``, ``semivariogram`` (estimated directly),
        ``implied_semivariogram`` (:math:`C(0) - C(h)`), ``sill``
        (:math:`C(0)`), ``n_pairs``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Secs. 1.4.2, 2.4.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    if coords.shape[0] != z.size:
        raise ValueError("`coords` and `z` must have the same number of rows")
    i, j = np.triu_indices(z.size, k=1)
    d = np.linalg.norm(coords[i] - coords[j], axis=1)
    mu = float(z.mean())
    cross = (z[i] - mu) * (z[j] - mu)
    if max_dist is None:
        max_dist = d.max() / 2.0 if d.size else 1.0
    keep = d <= max_dist
    d, cross = d[keep], cross[keep]
    edges = np.linspace(0.0, max_dist, int(n_bins) + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, int(n_bins) - 1)
    lag = np.full(int(n_bins), np.nan)
    cov = np.full(int(n_bins), np.nan)
    cnt = np.zeros(int(n_bins), dtype=int)
    for b in range(int(n_bins)):
        msk = idx == b
        cnt[b] = int(msk.sum())
        if cnt[b]:
            lag[b] = float(d[msk].mean())
            cov[b] = float(cross[msk].mean())
    sill = float(z.var(ddof=1))
    _, gam, _ = empirical_semivariogram(coords, z, n_bins, max_dist)
    return RichResult(
        title="Empirical covariance function",
        summary_lines=[("sill C(0)", sill), ("bins", int(n_bins))],
        payload={"lag": lag, "covariance": cov, "semivariogram": gam,
                 "implied_semivariogram": sill - cov, "sill": sill,
                 "n_pairs": cnt},
    )


def cheatsheet():
    return "spcovf: empirical C(h); gamma(h)=C(0)-C(h) only if 2nd-order stationary."
