"""Shared primitives for the Schabenberger & Gotway point-pattern chapter.

References
----------
Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
Spatial Data Analysis*. Chapman & Hall/CRC. Ch. 3.
"""

from . import _array_core as np

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
    # _array_core.minimum is a plain binary function, not a numpy ufunc, so
    # it has no .reduce; fold the four edge distances pairwise instead.
    return np.minimum(np.minimum(p[:, 0] - xmin, xmax - p[:, 0]),
                      np.minimum(p[:, 1] - ymin, ymax - p[:, 1]))


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

# --- Cross-K for bivariate point patterns, Sec. 3.4.4 -----------------------
#
#   eq (3.9)  Khat_ij(h) = [lam_i lam_j nu(A)]^-1
#                          sum_k sum_l w(s_k, u_l)^-1 I(h_kl <= h)
#
# with h_kl = ||s_k - u_l|| and w(s_k, u_l) "the proportion of the
# circumference of a circle centered at location s_k with radius h_kl that
# lies inside A" -- Ripley's isotropic edge correction.
#
# Lotwick and Silverman (1982), quoted on p. 104, note that although the
# population cross-K functions are symmetric under stationarity, the two
# estimators are not (Khat_12 != Khat_21), and give the more efficient
#
#   K*_ij(h) = {lam_j Khat_ij(h) + lam_i Khat_ji(h)} / (lam_i + lam_j)
#
# Under independence of the two processes K_ij(h) = pi h^2 regardless of the
# pattern of either type, so L*_ij(h) = sqrt(K*_ij(h)/pi) and L*_ij(h) - h is
# the diagnostic: positive means attraction, negative repulsion.
#
# Under the RANDOM LABELLING hypothesis of Diggle (1983) the relationship is
# instead eq (3.10), K_11 = K_22 = K_12, which Diggle and Chetwynd (1991) use
# to build a test on D(h) = K_ii(h) - K_jj(h). The book is emphatic that the
# two null hypotheses are different -- independence fixes the marginal
# structure of each process, random labelling conditions on all locations and
# randomises only the marks -- and that confusing them leads to "the analysis
# of data by methods which are largely irrelevant to the problem in hand".
# Both are provided, and named, so a caller has to choose one.


def ripley_weight(point, region, radius):
    """Proportion of the circle of radius ``radius`` centred at ``point``
    that lies inside the rectangular ``region``.

    Computed exactly rather than by sampling the circumference. A point
    ``point + radius*(cos t, sin t)`` is inside the rectangle exactly when

        -dx_min <= radius cos t <= dx_max
        -dy_min <= radius sin t <= dy_max

    Each bound switches status only at an angle where it is met with
    equality, so the circle splits into arcs at those critical angles and
    membership is constant on each arc. Summing the arcs whose midpoint
    satisfies all four constraints gives the exact proportion.
    """
    x, y = float(point[0]), float(point[1])
    xmin, ymin, xmax, ymax = region
    t = float(radius)
    if t <= 0:
        return 1.0
    dxm, dxp = x - xmin, xmax - x
    dym, dyp = y - ymin, ymax - y
    if min(dxm, dxp, dym, dyp) >= t:
        return 1.0                                   # circle wholly inside

    crit = [0.0, 2.0 * np.pi]
    for b in (-dxm / t, dxp / t):                    # cos t == b
        if -1.0 <= b <= 1.0:
            a = np.arccos(b)
            crit += [a, 2.0 * np.pi - a]
    for b in (-dym / t, dyp / t):                    # sin t == b
        if -1.0 <= b <= 1.0:
            a = np.arcsin(b)
            crit += [a % (2.0 * np.pi), (np.pi - a) % (2.0 * np.pi)]
    crit = np.unique(np.clip(np.asarray(crit, dtype=float), 0.0, 2.0 * np.pi))

    mid = 0.5 * (crit[:-1] + crit[1:])
    cx = x + t * np.cos(mid)
    cy = y + t * np.sin(mid)
    inside = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
    return float(np.diff(crit)[inside].sum() / (2.0 * np.pi))


def ripley_weights(points, region, radii):
    """Vectorised form of :func:`ripley_weight`.

    ``points`` is (N, 2) and ``radii`` is (N,), broadcast against each other;
    the result is (N,). Same critical-angle construction, done as array
    operations: every pair contributes at most eight critical angles plus the
    two endpoints, so the candidate set is a fixed (N, 10) block. Angles that
    do not exist for a given pair are filled with 0, which after sorting
    yields zero-length arcs and drops out of the sum.

    The scalar version is kept as the reference; ``schab_rest_verify`` checks
    the two agree exactly. Written because the estimator of eq (3.9) needs one
    weight per ordered pair, so a Python-level loop is O(n1 n2) calls -- at
    n = 1600 that is 2.6 million, which is minutes rather than seconds.
    """
    p = np.atleast_2d(np.asarray(points, dtype=float))
    t = np.atleast_1d(np.asarray(radii, dtype=float))
    xmin, ymin, xmax, ymax = region
    if p.shape[0] == 1 and t.size > 1:
        p = np.repeat(p, t.size, axis=0)
    n = t.size

    dxm = p[:, 0] - xmin
    dxp = xmax - p[:, 0]
    dym = p[:, 1] - ymin
    dyp = ymax - p[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_b = np.stack([-dxm / t, dxp / t], axis=1)      # (n, 2)
        sin_b = np.stack([-dym / t, dyp / t], axis=1)      # (n, 2)

    cand = np.zeros((n, 10))
    cand[:, 1] = 2.0 * np.pi
    ok_c = np.abs(cos_b) <= 1.0
    ac = np.arccos(np.clip(cos_b, -1.0, 1.0))
    cand[:, 2:4] = np.where(ok_c, ac, 0.0)
    cand[:, 4:6] = np.where(ok_c, 2.0 * np.pi - ac, 0.0)
    ok_s = np.abs(sin_b) <= 1.0
    as_ = np.arcsin(np.clip(sin_b, -1.0, 1.0))
    cand[:, 6:8] = np.where(ok_s, as_ % (2.0 * np.pi), 0.0)
    cand[:, 8:10] = np.where(ok_s, (np.pi - as_) % (2.0 * np.pi), 0.0)

    cand.sort(axis=1)
    mid = 0.5 * (cand[:, :-1] + cand[:, 1:])               # (n, 9)
    width = np.diff(cand, axis=1)
    cx = p[:, 0:1] + t[:, None] * np.cos(mid)
    cy = p[:, 1:2] + t[:, None] * np.sin(mid)
    inside = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
    w = (width * inside).sum(axis=1) / (2.0 * np.pi)

    # circles that never leave the window, and degenerate radii
    whole = np.minimum(np.minimum(dxm, dxp), np.minimum(dym, dyp)) >= t
    w = np.where(whole | (t <= 0), 1.0, w)
    return w


def cross_k_function(points1, points2, region, r, correction="ripley"):
    """eq (3.9): the cross-K estimator Khat_12(h).

    ``correction='ripley'`` uses the isotropic weight the book defines;
    ``'none'`` drops the weights, which biases the estimate downward near the
    boundary and is provided only so the effect of the correction is visible.
    """
    p1 = as_points(points1)
    p2 = as_points(points2)
    region = as_region(region, np.vstack([p1, p2]))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if np.any(r < 0):
        raise ValueError("`r` must be non-negative")
    if correction not in ("ripley", "none"):
        raise ValueError("`correction` must be 'ripley' or 'none'")
    n1, n2 = p1.shape[0], p2.shape[0]
    if n1 == 0 or n2 == 0:
        raise ValueError("both patterns must contain at least one event")
    area = region_area(region)
    lam1, lam2 = n1 / area, n2 / area

    d = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1)
    if correction == "none":
        winv = np.ones_like(d)
    else:
        pts = np.repeat(p1, n2, axis=0)
        w = ripley_weights(pts, region, d.ravel()).reshape(d.shape)
        winv = np.where(w > 0, 1.0 / np.where(w > 0, w, 1.0), 0.0)

    out = np.empty_like(r)
    for m, h in enumerate(r):
        out[m] = winv[d <= h].sum() / (lam1 * lam2 * area)
    return out


def cross_k_combined(points1, points2, region, r, correction="ripley"):
    """Lotwick and Silverman's pooled estimator, p. 104.

    ``K*_ij = {lam_j Khat_ij + lam_i Khat_ji} / (lam_i + lam_j)``

    Returns the pooled K, the two one-sided estimators, ``L* = sqrt(K*/pi)``
    and ``L* - h``, which is zero under independence.
    """
    p1 = as_points(points1)
    p2 = as_points(points2)
    region = as_region(region, np.vstack([p1, p2]))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    area = region_area(region)
    lam1, lam2 = p1.shape[0] / area, p2.shape[0] / area
    k12 = cross_k_function(p1, p2, region, r, correction)
    k21 = cross_k_function(p2, p1, region, r, correction)
    kstar = (lam2 * k12 + lam1 * k21) / (lam1 + lam2)
    lstar = np.sqrt(np.maximum(kstar, 0.0) / np.pi)
    return {
        "K_star": kstar,
        "K_12": k12,
        "K_21": k21,
        "L_star": lstar,
        "L_minus_h": lstar - r,
        "K_independence": np.pi * r ** 2,
        "r": r,
        "lambda_1": lam1,
        "lambda_2": lam2,
    }


def diggle_chetwynd_d(points1, points2, region, r, correction="border"):
    """``D(h) = K_ii(h) - K_jj(h)``, the random-labelling statistic.

    Diggle and Chetwynd (1991), via eq (3.10) ``K_11 = K_22 = K_12``. This
    tests a different null from :func:`cross_k_combined`: random labelling
    conditions on the full set of locations and randomises only the marks,
    whereas independence conditions on each process separately.
    """
    p1 = as_points(points1)
    p2 = as_points(points2)
    region = as_region(region, np.vstack([p1, p2]))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    k11 = k_function(p1, region, r, correction=correction)
    k22 = k_function(p2, region, r, correction=correction)
    return {"D": k11 - k22, "K_11": k11, "K_22": k22, "r": r}
