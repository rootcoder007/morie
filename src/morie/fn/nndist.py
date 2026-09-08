# morie.fn -- function file (rootcoder007/morie)
"""Nearest-neighbour distance distribution G(r)."""

import math

from . import _schab_pp as pp

from ._richresult import RichResult

__all__ = ["nearest_neighbor_distance"]


def nearest_neighbor_distance(coords, r_grid, window=None):
    """Border-corrected nearest-neighbour distance CDF.

    The raw empirical CDF of nearest-neighbour distances is biased
    downward near the edge of the window: a point close to the boundary
    may have its true nearest neighbour outside it, so the observed
    distance is too large.  The reduced-sample (border) correction
    removes exactly the points that could be affected at each radius,

        G_hat(r) = #{i : d_i <= r and b_i > r} / #{i : b_i > r},

    where ``d_i`` is the observed nearest-neighbour distance and
    ``b_i`` the distance from point i to the nearest window edge.  The
    complete spatial randomness benchmark is
    ``G_csr(r) = 1 - exp(-lambda pi r^2)``, so ``G_hat`` above it
    indicates clustering and below it regularity.

    Parameters
    ----------
    coords : array-like, shape (n, 2)
        Point coordinates.
    r_grid : array-like
        Non-negative radii at which to evaluate ``G``.
    window : sequence or None
        ``(xmin, ymin, xmax, ymax)``, or an (m, 2) vertex array whose
        bounding box is taken.  Defaults to the bounding box of
        ``coords``.

    Returns
    -------
    RichResult
        ``G`` (values on the grid), ``G_csr``, ``r`` (the grid),
        ``m_used`` (points retained per radius), ``estimate`` (the mean
        nearest-neighbour distance), ``lambda_hat``, ``n``.

    References
    ----------
    Diggle, P. J. (2003).  Statistical Analysis of Spatial Point
    Patterns, 2nd edition, Arnold; section 2.3 defines G and section
    4.3 the border correction.
    """
    p = pp.as_points(coords)
    n = len(p)
    if n < 2:
        raise ValueError("nearest_neighbor_distance: need at least two points")
    region = pp.as_region(window, p)
    lam = pp.intensity(p, region)
    d = [float(v) for v in pp.nn_distances(p)]
    b = [float(v) for v in pp.border_distance(p, region)]
    rs = [float(v) for v in (r_grid if hasattr(r_grid, "__len__") else [r_grid])]
    if not rs:
        raise ValueError("nearest_neighbor_distance: r_grid is empty")
    if any(v < 0.0 for v in rs):
        raise ValueError("nearest_neighbor_distance: r_grid must be non-negative")
    G, Gc, mu = [], [], []
    for h in rs:
        m = 0
        hit = 0
        for i in range(n):
            if b[i] > h:
                m += 1
                if d[i] <= h:
                    hit += 1
        mu.append(float(m))
        G.append(hit / m if m else float("nan"))
        Gc.append(1.0 - math.exp(-lam * math.pi * h * h))
    return RichResult(payload={
        "G": G, "G_csr": Gc, "r": rs, "m_used": mu,
        "estimate": sum(d) / n, "lambda_hat": lam, "n": n,
        "method": "Border-corrected nearest-neighbour distance CDF G(r)"})


def cheatsheet():
    return "nndist: Border-corrected nearest-neighbour distance CDF"


nearestneighbordistance = nearest_neighbor_distance
