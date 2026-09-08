"""Local-cluster first-order (nearest-neighbour) distance query.

Two questions get run together here because they answer each other.

The first is about the POINTS: are they clustered, spread out, or
consistent with complete spatial randomness? The first-order
nearest-neighbour distance is the classical way in -- under a
homogeneous Poisson process of intensity lambda the expected distance to
the nearest other point is 1 / (2 sqrt(lambda)), so the ratio of the
observed mean to that expectation is a scale-free index. Below one is
clustering, above one is regularity, and Clark and Evans' z statistic
says whether the departure is bigger than sampling noise.

The second is about a VALUE attached to the points: given a
neighbourhood defined by those same distances, how does the local mean
of x compare with the global one? The neighbourhood radius is not
invented -- it is the mean nearest-neighbour distance plus a multiple of
its standard deviation, so it is the scale the pattern itself sets. That
is what "first-order SD" names: one standard deviation out from the
first-order distance.

Edge effects are the thing that quietly ruins this. A point near the
boundary has neighbours outside the window that were never observed, so
its nearest-neighbour distance is too long and the index drifts towards
"regular". Three treatments, all selectable, and the choice travels in
the result:

  "none"      no correction. Honest for a window much larger than the
              typical spacing, wrong otherwise.
  "donnelly"  Donnelly's correction to the expectation and variance,
              which adds a perimeter term. Cheap, and the standard
              choice for a rectangular window.
  "buffer"    discard points within one neighbourhood radius of the
              boundary when computing the summary, but still allow them
              to serve as neighbours. This throws data away and is the
              most defensible: the retained points have complete
              neighbourhoods by construction.

Metrics: euclidean, manhattan, chebyshev. The metric is not decoration
-- on a street grid the Manhattan distance is the real one, and using
Euclidean there understates every distance by up to a factor of
sqrt(2).

References
  Clark, P.J. and Evans, F.C. (1954) "Distance to nearest neighbor as a
    measure of spatial relationships in populations." Ecology 35(4),
    445-453. The index and its z statistic.
  Donnelly, K. (1978) "Simulations to determine the variance and
    edge-effect of total nearest neighbour distance." In I. Hodder
    (ed.), Simulation Methods in Archaeology, Cambridge University
    Press, 91-95. The perimeter correction.
  Diggle, P.J. (2003) "Statistical Analysis of Spatial Point Patterns,"
    2nd edition. Arnold. Chapter 2: the G function and edge effects.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["lc_first_sd_query", "lcfsdq", "nn_distances", "clark_evans",
           "METRICS", "EDGE", "cheatsheet"]

METRICS = ("euclidean", "manhattan", "chebyshev")
EDGE = ("none", "donnelly", "buffer")


def _d(a, b, metric):
    if metric == "euclidean":
        return math.sqrt(_w.csum((a[k] - b[k]) * (a[k] - b[k])
                                 for k in range(len(a))))
    if metric == "manhattan":
        return _w.csum(abs(a[k] - b[k]) for k in range(len(a)))
    if metric == "chebyshev":
        return max(abs(a[k] - b[k]) for k in range(len(a)))
    raise ValueError("metric must be one of %r" % (METRICS,))


def nn_distances(coords, k=1, metric="euclidean"):
    """Distance from each point to its k-th nearest OTHER point.

    Ties are broken by index so the two arms select the same neighbour
    when two are equidistant -- which happens constantly on a lattice
    and would otherwise make the local statistics disagree while the
    distances matched.
    """
    n = len(coords)
    if k < 1 or k >= n:
        raise ValueError("k must lie in 1..n-1")
    out = []
    idx = []
    for i in range(n):
        pairs = sorted(((_d(coords[i], coords[j], metric), j)
                        for j in range(n) if j != i))
        out.append(pairs[k - 1][0])
        idx.append(pairs[k - 1][1])
    return out, idx


def _window(coords):
    """Bounding box area and perimeter, in the first two coordinates."""
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return w * h, 2.0 * (w + h), (min(xs), max(xs), min(ys), max(ys))


def clark_evans(dists, n, area, perimeter, edge="none"):
    """The Clark-Evans index and its z statistic.

    R = observed mean / expected mean, with the expectation
    1 / (2 sqrt(lambda)) under complete spatial randomness. Donnelly's
    correction adds a perimeter term to both the expectation and the
    variance, which is what stops a small window reading as regular.
    """
    if edge not in EDGE:
        raise ValueError("edge must be one of %r" % (EDGE,))
    if area <= 0.0:
        raise ValueError("the window has zero area")
    lam = n / area
    obs = _w.csum(dists) / len(dists)
    exp_d = 0.5 / math.sqrt(lam)
    var_d = (4.0 - math.pi) / (4.0 * math.pi * lam * n)
    if edge == "donnelly":
        exp_d = (0.5 * math.sqrt(area / n)
                 + (0.0514 + 0.041 / math.sqrt(n)) * perimeter / n)
        var_d = (0.0703 * area / (n * n)
                 + 0.037 * perimeter * math.sqrt(area / (n ** 5)))
    se = math.sqrt(var_d) if var_d > 0.0 else float("nan")
    z = (obs - exp_d) / se if se > 0.0 else float("nan")
    return {"R": obs / exp_d if exp_d > 0.0 else float("nan"),
            "observed": obs, "expected": exp_d, "se": se, "z": z,
            "p": 2.0 * (1.0 - _w.ncdf(abs(z))) if z == z else float("nan"),
            "lambda": lam, "edge": edge}


def lc_first_sd_query(x, coords, k=1, metric="euclidean", edge="none",
                      sd_multiplier=1.0, area=None, perimeter=None,
                      grid=None):
    """Nearest-neighbour summary of a pattern and a local query on x.

    Parameters
    ----------
    x : sequence
        A value attached to each point. Pass a constant if only the
        pattern is of interest.
    coords : sequence of sequences
        Point coordinates. The first two are taken as the plane for
        the window; the distance uses all of them.
    k : int
        Which nearest neighbour to use for the first-order distance.
    metric : str
        A member of METRICS.
    edge : str
        A member of EDGE.
    sd_multiplier : float
        The neighbourhood radius is mean(d) + this times sd(d).
    area, perimeter : float or None
        The window. Taken from the bounding box when omitted, which
        UNDERSTATES the true window whenever the points do not reach
        its edges -- so the index reads as more clustered than it is,
        and passing the real window is worth doing.
    grid : sequence or None
        Radii at which the empirical G function is reported.

    Returns
    -------
    RichResult
        The nearest-neighbour distances, the Clark-Evans index and test,
        the neighbourhood radius, per-point local means and counts of x,
        the points flagged as locally clustered or isolated, and the G
        function against its complete-spatial-randomness expectation.

    References
    ----------
    Clark and Evans (1954) Ecology 35(4), 445-453; Donnelly (1978) in
    Hodder (ed.) Simulation Methods in Archaeology, 91-95; Diggle (2003)
    ch. 2.
    """
    if metric not in METRICS:
        raise ValueError("metric must be one of %r" % (METRICS,))
    if edge not in EDGE:
        raise ValueError("edge must be one of %r" % (EDGE,))
    pts = [[float(v) for v in p] for p in coords]
    n = len(pts)
    if n < 3:
        raise ValueError("need at least three points")
    if len(pts[0]) < 2:
        raise ValueError("coordinates must have at least two dimensions")
    xv = [float(v) for v in x]
    if len(xv) != n:
        raise ValueError("x and coords must have the same length")

    dists, nbr = nn_distances(pts, k, metric)
    mean_d = _w.csum(dists) / n
    sd_d = math.sqrt(_w.csum((v - mean_d) * (v - mean_d) for v in dists)
                     / (n - 1))
    radius = mean_d + float(sd_multiplier) * sd_d

    box_area, box_perim, bb = _window(pts)
    A = box_area if area is None else float(area)
    P = box_perim if perimeter is None else float(perimeter)

    if edge == "buffer":
        keep = [i for i in range(n)
                if (pts[i][0] - bb[0] >= radius and bb[1] - pts[i][0] >= radius
                    and pts[i][1] - bb[2] >= radius
                    and bb[3] - pts[i][1] >= radius)]
        if len(keep) < 3:
            raise ValueError("the buffer left fewer than three points; "
                             "use a smaller sd_multiplier or another "
                             "edge correction")
        ce = clark_evans([dists[i] for i in keep], n, A, P, "none")
        ce["edge"] = "buffer"
        ce["n_kept"] = len(keep)
    else:
        keep = list(range(n))
        ce = clark_evans(dists, n, A, P, edge)
        ce["n_kept"] = n

    gmean = _w.csum(xv) / n
    gsd = math.sqrt(_w.csum((v - gmean) * (v - gmean) for v in xv)
                    / (n - 1)) if n > 1 else 0.0

    local_mean = []
    local_count = []
    local_z = []
    for i in range(n):
        members = [j for j in range(n)
                   if j != i and _d(pts[i], pts[j], metric) <= radius]
        if members:
            lm = _w.csum(xv[j] for j in members) / len(members)
        else:
            lm = float("nan")
        local_mean.append(lm)
        local_count.append(len(members))
        # A z against the sampling distribution of a mean of that many
        # draws, which is the only comparison that is fair to a
        # neighbourhood of two and one of twenty at the same time.
        if members and gsd > 0.0:
            local_z.append((lm - gmean) / (gsd / math.sqrt(len(members))))
        else:
            local_z.append(float("nan"))

    clustered = [i for i in range(n)
                 if dists[i] < mean_d - float(sd_multiplier) * sd_d]
    isolated = [i for i in range(n)
                if dists[i] > mean_d + float(sd_multiplier) * sd_d]

    if grid is None:
        grid = [radius * (t + 1) / 8.0 for t in range(8)]
    grid = [float(v) for v in grid]
    lam = n / A
    gfun = []
    gcsr = []
    for r in grid:
        gfun.append(_w.csum(1.0 for v in dists if v <= r) / n)
        # The CSR expectation for the nearest-neighbour distance, which
        # is a two-dimensional statement: an area, not a length.
        gcsr.append(1.0 - math.exp(-lam * math.pi * r * r))

    return RichResult(payload={
        "nn_distance": dists,
        "nn_index": nbr,
        "mean_nn": mean_d,
        "sd_nn": sd_d,
        "radius": radius,
        "clark_evans": ce,
        "R": ce["R"],
        "z": ce["z"],
        "p": ce["p"],
        "local_mean": local_mean,
        "local_count": local_count,
        "local_z": local_z,
        "clustered": clustered,
        "isolated": isolated,
        "n_clustered": len(clustered),
        "n_isolated": len(isolated),
        "grid": grid,
        "G": gfun,
        "G_csr": gcsr,
        "area": A,
        "perimeter": P,
        "global_mean": gmean,
        "global_sd": gsd,
        "n": n,
        "k": int(k),
        "metric": metric,
        "edge": edge,
        "estimate": ce["R"],
        "se": ce["se"],
        "method": "first-order nearest-neighbour cluster query",
    })


lcfsdq = lc_first_sd_query


def cheatsheet():
    return ("lcfsdq: first-order nearest-neighbour cluster query. "
            "metrics " + ", ".join(METRICS) + "; edge " + ", ".join(EDGE))


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
lcfirstsdquery = lc_first_sd_query
