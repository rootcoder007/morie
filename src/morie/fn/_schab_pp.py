"""Shared primitives for the Schabenberger & Gotway point-pattern chapter.

References
----------
Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
Spatial Data Analysis*. Chapman & Hall/CRC. Ch. 3.
"""

import numpy as np

__all__ = []


def as_points(points):
    p = np.atleast_2d(np.asarray(points, dtype=float))
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("`points` must be an (n, 2) array of coordinates")
    return p


def as_region(region, points=None):
    """Return ``(xmin, ymin, xmax, ymax)``.

    Accepts a 4-sequence bounding box, or an (m, 2) polygon/point set
    whose bounding box is taken. ``None`` falls back to the bounding box
    of ``points``.
    """
    if region is None:
        if points is None:
            raise ValueError("`region` is required when `points` is not given")
        p = as_points(points)
        return (float(p[:, 0].min()), float(p[:, 1].min()),
                float(p[:, 0].max()), float(p[:, 1].max()))
    r = np.asarray(region, dtype=float)
    if r.ndim == 1 and r.size == 4:
        xmin, ymin, xmax, ymax = (float(v) for v in r)
    elif r.ndim == 2 and r.shape[1] >= 2:
        xmin, ymin = float(r[:, 0].min()), float(r[:, 1].min())
        xmax, ymax = float(r[:, 0].max()), float(r[:, 1].max())
    else:
        raise ValueError("`region` must be (xmin, ymin, xmax, ymax) or an "
                         "(m, 2) array of vertices")
    if not (xmax > xmin and ymax > ymin):
        raise ValueError("`region` must have positive area")
    return xmin, ymin, xmax, ymax


def region_area(region):
    xmin, ymin, xmax, ymax = region
    return (xmax - xmin) * (ymax - ymin)


def intensity(points, region):
    """lambda_hat = N(A) / nu(A), eq (3.8)."""
    return as_points(points).shape[0] / region_area(region)


def pair_distances(points):
    p = as_points(points)
    i, j = np.triu_indices(p.shape[0], k=1)
    return np.linalg.norm(p[i] - p[j], axis=1)


def nn_distances(points):
    """Nearest-neighbour distance h_i for each event."""
    p = as_points(points)
    n = p.shape[0]
    if n < 2:
        return np.zeros(0)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def border_distance(points, region):
    """Distance from each event to the nearest edge of the bounding region."""
    p = as_points(points)
    xmin, ymin, xmax, ymax = region
    return np.minimum.reduce([p[:, 0] - xmin, xmax - p[:, 0],
                              p[:, 1] - ymin, ymax - p[:, 1]])


def k_function(points, region, r, correction="border"):
    """Ripley's K, estimated as in Sec. 3.4.2.

    Naive moment estimator (no correction):

        E_tilde(h) = (1/n) sum_i sum_{j != i} I(h_ij <= h)
        K_tilde(h) = E_tilde(h) / lambda_hat

    Border correction ("reduced sample") uses only events further than h
    from the boundary, which removes the negative bias from events whose
    neighbours fall outside the observation window.
    """
    p = as_points(points)
    n = p.shape[0]
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if np.any(r < 0):
        raise ValueError("`r` must be non-negative")
    lam = intensity(p, region)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    out = np.empty_like(r)
    if correction == "none":
        for k, h in enumerate(r):
            out[k] = (d <= h).sum() / n / lam
        return out
    if correction != "border":
        raise ValueError("`correction` must be 'border' or 'none'")
    db = border_distance(p, region)
    for k, h in enumerate(r):
        keep = db > h
        m = int(keep.sum())
        out[k] = ((d[keep] <= h).sum() / m / lam) if m else np.nan
    return out
