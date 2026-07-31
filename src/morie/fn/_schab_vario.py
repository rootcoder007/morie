"""Shared primitives for the Schabenberger & Gotway variogram family.

One implementation per book equation, so the individual model modules
stay thin and cannot drift apart.

References
----------
Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
Spatial Data Analysis*. Chapman & Hall/CRC. Sec. 4.3.
"""

import numpy as np

__all__ = []

# The book parameterises the exponential and gaussian models by the
# PRACTICAL range alpha -- the lag at which correlation has fallen to
# exp(-3) = 0.049787..., i.e. "0.05 or less" (p. 143, eqs 4.10-4.11).
PRACTICAL_RANGE_C = 3.0


def _as_lag(h):
    h = np.atleast_1d(np.asarray(h, dtype=float))
    if np.any(h < 0):
        raise ValueError("lag distances `h` must be non-negative")
    return h


def _check_params(nugget, sill, rng):
    if nugget < 0:
        raise ValueError("`nugget` must be >= 0")
    if sill < 0:
        raise ValueError("`sill` (partial sill) must be >= 0")
    if rng <= 0:
        raise ValueError("`range` must be > 0")


def correlogram(h, rng, model):
    """R(h) for the isotropic models of Sec. 4.3, on the practical-range scale."""
    h = _as_lag(h)
    if model == "exponential":                       # eq (4.11)
        return np.exp(-PRACTICAL_RANGE_C * h / rng)
    if model == "gaussian":                          # eq (4.10)
        return np.exp(-PRACTICAL_RANGE_C * (h / rng) ** 2)
    if model == "spherical":                         # eq (4.13)
        r = np.zeros_like(h)
        inside = h <= rng
        u = h[inside] / rng
        r[inside] = 1.0 - 1.5 * u + 0.5 * u**3
        return r
    raise ValueError(f"unknown model {model!r}")


def semivariogram(h, nugget, sill, rng, model):
    """gamma(h) = c0 + sigma0^2 (1 - R(h)) for h > 0, and gamma(0) = 0.

    The nugget is a discontinuity AT the origin: gamma(0) = 0 by
    definition even when c0 > 0 (Sec. 4.3.6).
    """
    _check_params(nugget, sill, rng)
    h = _as_lag(h)
    g = nugget + sill * (1.0 - correlogram(h, rng, model))
    g[h == 0] = 0.0
    return g


def empirical_semivariogram(coords, z, n_bins=15, max_dist=None):
    """Matheron's method-of-moments estimator, eq (4.1).

    gamma_hat(h) = 1 / (2 |N(h)|) * sum over N(h) of (Z(s_i) - Z(s_j))^2
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    if coords.shape[0] != z.size:
        raise ValueError("`coords` and `z` must have the same number of rows")
    i, j = np.triu_indices(z.size, k=1)
    d = np.linalg.norm(coords[i] - coords[j], axis=1)
    sq = (z[i] - z[j]) ** 2
    if max_dist is None:
        max_dist = d.max() / 2.0 if d.size else 1.0
    keep = d <= max_dist
    d, sq = d[keep], sq[keep]
    edges = np.linspace(0.0, max_dist, int(n_bins) + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, int(n_bins) - 1)
    lag = np.full(int(n_bins), np.nan)
    gam = np.full(int(n_bins), np.nan)
    cnt = np.zeros(int(n_bins), dtype=int)
    for b in range(int(n_bins)):
        m = idx == b
        cnt[b] = int(m.sum())
        if cnt[b]:
            lag[b] = float(d[m].mean())
            gam[b] = float(sq[m].sum() / (2.0 * cnt[b]))
    return lag, gam, cnt
