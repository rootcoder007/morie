"""Empirical (Matheron) variogram gamma(h)."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_vario import empirical_semivariogram

__all__ = ["vgrm", "variogram"]


def vgrm(coords, values, bins=15):
    """
    Matheron method-of-moments empirical semivariogram.

        gamma_hat(h_k) = 1 / (2 |N(h_k)|) * sum_{(i,j) in N(h_k)} (Z(s_i) - Z(s_j))^2

    where N(h_k) collects the pairs whose separation distance falls in
    lag bin k.

    Sources
    -------
    Matheron, G. (1963). Principles of geostatistics. *Economic Geology*,
    58(8), 1246-1266 (the estimator originates here).
    Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
    Spatial Data Analysis*, Chapman & Hall/CRC, Sec. 4.2, eq. (4.1)
    (local PDF: WD_BLACK/library/pdf/Statistical_Methods_for_Spatial_Data_Analysis.pdf).
    Implementation delegates the equal-width path to the verified
    ``_schab_vario.empirical_semivariogram`` (same eq. 4.1).

    Parameters
    ----------
    coords : array-like, (n, d)
        Site coordinates.
    values : array-like, (n,)
        Observations Z(s_i).
    bins : int or array-like
        Number of equal-width lag bins up to half the maximum pairwise
        distance (int), or explicit bin edges (array-like, ascending).

    Returns
    -------
    RichResult
        Keys: lag (mean pair distance per bin), gamma, n_pairs, edges.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    values = np.asarray(values, dtype=float).ravel()
    if coords.shape[0] != values.size:
        raise ValueError("`coords` and `values` must have the same number of rows")
    if not hasattr(bins, "__len__"):
        n_bins = int(bins)
        if n_bins < 1:
            raise ValueError("`bins` must be a positive integer or explicit edges")
        lag, gam, cnt = empirical_semivariogram(coords, values, n_bins=n_bins)
        i, j = np.triu_indices(values.size, k=1)
        d = np.linalg.norm(coords[i] - coords[j], axis=1)
        max_dist = d.max() / 2.0 if d.size else 1.0
        edges = np.linspace(0.0, max_dist, n_bins + 1)
    else:
        edges = np.asarray(bins, dtype=float).ravel()
        if edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("explicit `bins` edges must be ascending with >= 2 entries")
        n_bins = edges.size - 1
        i, j = np.triu_indices(values.size, k=1)
        d = np.linalg.norm(coords[i] - coords[j], axis=1)
        sq = (values[i] - values[j]) ** 2
        keep = (d >= edges[0]) & (d <= edges[-1])
        d, sq = d[keep], sq[keep]
        idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
        lag = np.full(n_bins, np.nan)
        gam = np.full(n_bins, np.nan)
        cnt = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            m = idx == b
            cnt[b] = int(m.sum())
            if cnt[b]:
                lag[b] = float(d[m].mean())
                gam[b] = float(sq[m].sum() / (2.0 * cnt[b]))
    return RichResult(payload={
        "lag": lag, "gamma": gam, "n_pairs": cnt, "edges": edges,
        "n": int(values.size),
        "method": "Matheron empirical semivariogram (Schabenberger-Gotway eq. 4.1)",
    })


# long descriptive alias (stub-era name)
variogram = vgrm


def cheatsheet():
    return "vgrm: Matheron empirical variogram gamma(h), eq. 4.1 Schabenberger-Gotway"
