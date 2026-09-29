# morie.fn -- function file (rootcoder007/morie)
"""Geodesy and spatial sampling: rhumb-line (loxodrome) distance and bearing, geodetic/ECEF
conversion and the seven-parameter Helmert datum transformation, spatial block and k-means
cross-validation folds, temporally stratified sampling and nested (quadtree) multiresolution
grid indexing."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = [
    "rhumb_line",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "helmert_transform",
    "block_cv_folds",
    "kmeans_cv_folds",
    "temporal_stratified_sample",
    "nested_grid",
]

_ELLIPSOIDS = {
    "WGS84": (6378137.0, 1 / 298.257223563),
    "GRS80": (6378137.0, 1 / 298.257222101),
    "WGS72": (6378135.0, 1 / 298.26),
    "Clarke1866": (6378206.4, 1 / 294.9786982),
    "Airy1830": (6377563.396, 1 / 299.3249646),
    "Intl1924": (6378388.0, 1 / 297.0),
}


def rhumb_line(lat1, lon1, lat2, lon2, *, radius: float = 6378137.0, method: str = "shortest") -> RichResult:
    r"""Rhumb-line (constant-bearing, loxodrome) distance and bearing on a sphere.

    ``dpsi = ln(tan(pi/4 + phi2/2) / tan(pi/4 + phi1/2))`` (Mercator
    stretched latitude difference), ``q = dphi / dpsi`` (``cos phi`` on an
    E-W line), ``d = R sqrt(dphi^2 + q^2 dlambda^2)`` and bearing
    ``atan2(dlambda, dpsi)``; the longitude difference takes the shorter way
    round (as ``geosphere::distRhumb``). ``method="geosphere"`` reproduces
    ``geosphere::bearingRhumb``, which wraps an antimeridian-crossing
    longitude difference twice and so reports the bearing of the long way
    round (the distance is unaffected).

    References
    ----------
    Williams, E. (2011). Aviation Formulary v1.47. Bowditch, N. (2002). The
    American Practical Navigator, ch. 24.

    Examples
    --------
    >>> r = rhumb_line(0.0, 0.0, 0.0, 1.0)
    >>> round(r.distance, 6), round(r.bearing, 12)
    (111319.490793, 90.0)
    """
    rad = math.pi / 180.0
    p1, p2 = lat1 * rad, lat2 * rad
    dphi = p2 - p1
    dl0 = (lon2 - lon1) * rad
    dl = dl0
    if abs(dl) > math.pi:
        dl = -(2 * math.pi - dl) if dl > 0 else 2 * math.pi + dl
    db = dl
    if method == "geosphere" and abs(dl0) > math.pi and dl0 > 0:
        db = 2 * math.pi + dl
    dpsi = math.log(math.tan(math.pi / 4 + p2 / 2) / math.tan(math.pi / 4 + p1 / 2))
    q = dphi / dpsi if abs(dpsi) > 1e-12 else math.cos(p1)
    d = math.sqrt(dphi * dphi + q * q * dl * dl) * radius
    brg = (math.atan2(db, dpsi) / rad) % 360.0
    return RichResult(payload={"distance": d, "bearing": brg})


def geodetic_to_ecef(lat, lon, h, *, ellipsoid="WGS84") -> list:
    r"""Geodetic latitude, longitude (degrees) and height to Earth-centred Cartesian ``X, Y, Z``.

    ``N = a / sqrt(1 - e^2 sin^2 phi)``, ``X = (N + h) cos phi cos lambda``,
    ``Y = (N + h) cos phi sin lambda``, ``Z = (N (1 - e^2) + h) sin phi``.

    References
    ----------
    IOGP (2019). Geomatics Guidance Note 7-2: Coordinate Conversions and
    Transformations including Formulas, section 2.2.1.

    Examples
    --------
    >>> [round(v, 6) for v in geodetic_to_ecef(0.0, 0.0, 0.0)]
    [6378137.0, 0.0, 0.0]
    """
    a, f = _ELLIPSOIDS[ellipsoid] if isinstance(ellipsoid, str) else ellipsoid
    e2 = f * (2 - f)
    p, lm = lat * math.pi / 180, lon * math.pi / 180
    N = a / math.sqrt(1 - e2 * math.sin(p) ** 2)
    return [
        (N + h) * math.cos(p) * math.cos(lm),
        (N + h) * math.cos(p) * math.sin(lm),
        (N * (1 - e2) + h) * math.sin(p),
    ]


def ecef_to_geodetic(X, Y, Z, *, ellipsoid="WGS84", tol: float = 1e-14) -> list:
    r"""Earth-centred ``X, Y, Z`` to geodetic latitude, longitude (degrees) and height (Bowring iteration).

    ``lambda = atan2(Y, X)``; ``phi`` iterates
    ``phi = atan((Z + e^2 N sin phi) / p)`` to convergence, ``h = p / cos phi - N``.

    Examples
    --------
    >>> [round(v, 9) for v in ecef_to_geodetic(*geodetic_to_ecef(45.0, 10.0, 100.0))]
    [45.0, 10.0, 100.0]
    """
    a, f = _ELLIPSOIDS[ellipsoid] if isinstance(ellipsoid, str) else ellipsoid
    e2 = f * (2 - f)
    p = math.hypot(X, Y)
    lon = math.atan2(Y, X)
    phi = math.atan2(Z, p * (1 - e2))
    for _ in range(50):
        N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
        new = math.atan2(Z + e2 * N * math.sin(phi), p)
        if abs(new - phi) <= tol:
            phi = new
            break
        phi = new
    N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    h = p / math.cos(phi) - N if abs(math.cos(phi)) > 1e-12 else abs(Z) - a * math.sqrt(1 - e2)
    return [phi * 180 / math.pi, lon * 180 / math.pi, h]


def helmert_transform(
    xyz,
    tx: float,
    ty: float,
    tz: float,
    rx: float,
    ry: float,
    rz: float,
    ds_ppm: float,
    *,
    convention: str = "position_vector",
) -> list:
    r"""Seven-parameter Helmert (similarity) datum transformation of ECEF coordinates.

    ``X' = T + (1 + ds) R X`` with rotations in arc-seconds and scale in
    parts per million; the position-vector convention (EPSG 1033) uses
    ``R = [[1, -rz, ry], [rz, 1, -rx], [-ry, rx, 1]]`` and the
    coordinate-frame convention (EPSG 1032) the transpose (small-angle form).

    References
    ----------
    IOGP (2019). Geomatics Guidance Note 7-2, section 4.4.3. Helmert, F. R.
    (1880). Die mathematischen und physikalischen Theorieen der hoeheren Geodaesie.

    Examples
    --------
    >>> [round(v, 1) for v in helmert_transform([3657660.66, 255768.55, 5201382.11], 0, 0, 4.5, 0, 0, 0.554, 0.219)]
    [3657660.8, 255778.4, 5201387.7]
    """
    s = 1.0 + ds_ppm * 1e-6
    k = math.pi / (180 * 3600)
    a, b, c = rx * k, ry * k, rz * k
    if convention == "coordinate_frame":
        a, b, c = -a, -b, -c
    R = [[1.0, -c, b], [c, 1.0, -a], [-b, a, 1.0]]
    x = [float(v) for v in xyz]
    return [t + s * ssum(R[i][j] * x[j] for j in range(3)) for i, t in enumerate((tx, ty, tz))]


def block_cv_folds(coords, nx: int, ny: int, k: int, *, seed: int = 0) -> RichResult:
    r"""Spatial block cross-validation folds (Roberts et al. 2017).

    The bounding box is cut into ``nx x ny`` equal blocks (a point on the
    upper edge belongs to the last block); the non-empty blocks are shuffled
    by a Philox-driven Fisher-Yates permutation and dealt to ``k`` folds in
    turn, and each point inherits its block's fold (0-based).

    References
    ----------
    Roberts, D. R. et al. (2017). Cross-validation strategies for data with
    temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40,
    913-929. Valavi, R. et al. (2019). blockCV. Methods Ecol. Evol. 10, 225-232.

    Examples
    --------
    >>> r = block_cv_folds([(0, 0), (1, 0), (0, 1), (1, 1)], 2, 2, 2)
    >>> sorted(r.fold)
    [0, 0, 1, 1]
    """
    xs = [float(c[0]) for c in coords]
    ys = [float(c[1]) for c in coords]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    def cell(v, lo, hi, m):
        if hi <= lo:
            return 0
        return min(int(math.floor((v - lo) / (hi - lo) * m)), m - 1)

    block = [cell(x, x0, x1, nx) + nx * cell(y, y0, y1, ny) for x, y in zip(xs, ys)]
    used = sorted(set(block))
    u = random_uniform(len(used), seed=seed, stream=0) if used else []
    perm = list(used)
    for i in range(len(perm) - 1, 0, -1):
        j = int(math.floor(float(u[i]) * (i + 1)))
        perm[i], perm[j] = perm[j], perm[i]
    fold_of = {b: q % k for q, b in enumerate(perm)}
    return RichResult(payload={"fold": [fold_of[b] for b in block], "block": block})


def kmeans_cv_folds(coords, k: int, *, max_iter: int = 100, seed: int = 0) -> RichResult:
    r"""Spatial cross-validation folds from k-means clusters of the coordinates (Brenning 2012).

    Lloyd's algorithm from k-means++ seeding (Arthur and Vassilvitskii 2007)
    driven by the Philox stream of ``seed``; each cluster is a fold (0-based,
    ordered by the first point of each cluster).

    References
    ----------
    Brenning, A. (2012). Spatial cross-validation and bootstrap for the
    assessment of prediction rules in remote sensing: the R package
    sperrorest. IGARSS 2012, 5372-5375. Arthur, D. and Vassilvitskii, S.
    (2007). k-means++: the advantages of careful seeding. SODA 2007, 1027-1035.

    Examples
    --------
    >>> r = kmeans_cv_folds([(0, 0), (0.1, 0), (5, 5), (5.1, 5)], 2)
    >>> r.fold
    [0, 0, 1, 1]
    """
    P = [(float(a), float(b)) for a, b in coords]
    n = len(P)
    u = random_uniform(k, seed=seed, stream=0)
    first = min(int(math.floor(float(u[0]) * n)), n - 1)
    cent = [P[first]]
    for c in range(1, k):
        d2 = [min((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in cent) for p in P]
        tot = ssum(d2)
        target = float(u[c]) * tot
        acc, pick = 0.0, n - 1
        for i, v in enumerate(d2):
            acc += v
            if acc >= target:
                pick = i
                break
        cent.append(P[pick])
    lab = [0] * n
    for _ in range(max_iter):
        new = [min(range(k), key=lambda j: ((p[0] - cent[j][0]) ** 2 + (p[1] - cent[j][1]) ** 2, j)) for p in P]
        cent = [
            (
                ssum(P[i][0] for i in range(n) if new[i] == j) / max(1, new.count(j)),
                ssum(P[i][1] for i in range(n) if new[i] == j) / max(1, new.count(j)),
            )
            if j in new
            else cent[j]
            for j in range(k)
        ]
        if new == lab:
            break
        lab = new
    order = {}
    for v in lab:
        if v not in order:
            order[v] = len(order)
    return RichResult(payload={"fold": [order[v] for v in lab], "centers": cent})


def temporal_stratified_sample(times, n_sample: int, *, n_strata: int = 4, seed: int = 0) -> RichResult:
    r"""Temporally stratified random sample with proportional allocation.

    Times are cut into ``n_strata`` equal-width periods; stratum ``h`` of
    size ``N_h`` gets ``n_h`` units by largest-remainder rounding of
    ``n N_h / N`` and a simple random sample without replacement (Philox
    stream ``h``). Returns 0-based indices and the stratum weights ``N_h / n_h``.

    References
    ----------
    Cochran, W. G. (1977). Sampling Techniques, 3rd ed., ch. 5. de Gruijter,
    J. et al. (2006). Sampling for Natural Resource Monitoring, ch. 14.

    Examples
    --------
    >>> r = temporal_stratified_sample(list(range(8)), 4, n_strata=2)
    >>> r.allocation
    [2, 2]
    """
    ts = [float(v) for v in times]
    N = len(ts)
    lo, hi = min(ts), max(ts)
    st = [min(int(math.floor((v - lo) / (hi - lo) * n_strata)), n_strata - 1) if hi > lo else 0 for v in ts]
    sizes = [st.count(h) for h in range(n_strata)]
    raw = [n_sample * s / N for s in sizes]
    alloc = [int(math.floor(v)) for v in raw]
    rem = sorted(range(n_strata), key=lambda h: (-(raw[h] - alloc[h]), h))
    for h in rem[: n_sample - sum(alloc)]:
        alloc[h] += 1
    chosen, weights = [], []
    for h in range(n_strata):
        idx = [i for i in range(N) if st[i] == h]
        u = random_uniform(len(idx), seed=seed, stream=h) if idx else []
        pool = list(idx)
        for i in range(len(pool) - 1, 0, -1):
            j = int(math.floor(float(u[i]) * (i + 1)))
            pool[i], pool[j] = pool[j], pool[i]
        pick = sorted(pool[: alloc[h]])
        chosen += pick
        weights += [sizes[h] / alloc[h]] * len(pick)
    return RichResult(payload={"index": chosen, "weight": weights, "allocation": alloc, "stratum_sizes": sizes})


def nested_grid(coords, levels: int, *, bbox=None) -> RichResult:
    r"""Nested multiresolution (quadtree) grid indexing of points.

    At level ``l`` (0 = coarsest single cell) the box is split into
    ``2^l x 2^l`` cells; each point gets its cell index
    ``i_x + 2^l i_y`` at every level (the parent of a cell at level ``l`` is
    the cell containing it at level ``l - 1``), and cell counts are returned
    per level.

    References
    ----------
    Samet, H. (1984). The quadtree and related hierarchical data structures.
    ACM Computing Surveys 16, 187-260.

    Examples
    --------
    >>> r = nested_grid([(0.1, 0.1), (0.9, 0.9), (0.6, 0.2)], 2, bbox=(0, 1, 0, 1))
    >>> r.cells[1], r.cells[2]
    ([0, 3, 1], [0, 15, 2])
    """
    P = [(float(a), float(b)) for a, b in coords]
    if bbox is None:
        bbox = (min(p[0] for p in P), max(p[0] for p in P), min(p[1] for p in P), max(p[1] for p in P))
    x0, x1, y0, y1 = bbox
    cells, counts = [], []
    for lev in range(levels + 1):
        m = 2**lev
        ids = []
        for x, y in P:
            ix = min(int(math.floor((x - x0) / (x1 - x0) * m)), m - 1) if x1 > x0 else 0
            iy = min(int(math.floor((y - y0) / (y1 - y0) * m)), m - 1) if y1 > y0 else 0
            ids.append(ix + m * iy)
        cells.append(ids)
        counts.append([ids.count(c) for c in range(m * m)])
    return RichResult(payload={"cells": cells, "counts": counts})


def cheatsheet() -> str:
    return (
        "rhumb_line / geodetic_to_ecef / ecef_to_geodetic / helmert_transform / block_cv_folds / kmeans_cv_folds / "
        "temporal_stratified_sample / nested_grid -> geodesy and spatial sampling."
    )
