# morie.fn -- function file (rootcoder007/morie)
"""Variogram for lattice data."""

from __future__ import annotations

from . import _lattice as lat
from ._containers import SpatialResult


def lacvgm(y, coords, n_lags=15, cutoff=None):
    r"""Variogram for lattice data.

    Matheron's (1962) method-of-moments semivariogram of values attached to
    the lattice units' reference points: ``gamma(h_k) = sum_{N(h_k)} (y_i -
    y_j)^2 / (2 |N(h_k)|)`` over the pairs whose distance falls in bin
    ``(k w, (k + 1) w]``, ``w = cutoff / n_lags`` (Cressie 1993, sec. 2.4).
    Defaults as ``gstat::variogram``: ``cutoff`` one third of the
    bounding-box diagonal and 15 bins; ``dist`` is the mean pair distance of
    each non-empty bin.

    Parameters
    ----------
    y : array-like, shape (n,)
        Values.
    coords : array-like, shape (n, 2)
        Unit coordinates (e.g. centroids).
    n_lags : int
        Number of distance bins.
    cutoff : float, optional
        Largest distance considered.

    Returns
    -------
    SpatialResult
        ``statistic`` is the semivariance of the first non-empty bin;
        ``extra`` has ``np``, ``dist``, ``gamma``, ``cutoff``, ``width``.

    References
    ----------
    Matheron, G. (1962). *Traite de geostatistique appliquee*. Technip, Paris.

    Cressie, N. (1993). *Statistics for Spatial Data*, revised edition. Wiley, New York, sec. 2.4.

    Examples
    --------
    >>> xy = [[0, 0], [1, 0.2], [2.1, 0], [0.1, 1], [1.2, 1.1], [2, 0.9]]
    >>> r = lacvgm([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], xy, n_lags=3, cutoff=2.0)
    >>> r.extra["np"], [round(g, 12) for g in r.extra["gamma"]]
    ([9, 3], [0.594444444444, 0.33])
    """
    r = lat.semivariogram(lat.vec(y), lat.mat(coords), int(n_lags), cutoff)
    return SpatialResult(name="lacvgm", statistic=r["gamma"][0] if r["gamma"] else float("nan"), extra=r)


lacvgm_fn = lacvgm


def cheatsheet() -> str:
    return "lacvgm(y, coords, n_lags, cutoff) -> Matheron semivariogram (gstat defaults)"
