# morie.fn -- function file (rootcoder007/morie)
"""Digital elevation model analysis: terrain derivatives, D8/D-infinity/MFD flow routing, sinks, streams, viewsheds."""

from __future__ import annotations

import heapq
import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "terrain_indices",
    "hillshade",
    "d8_flow_direction",
    "flow_accumulation",
    "watershed",
    "fill_sinks",
    "stream_order",
    "topographic_wetness_index",
    "stream_power_index",
    "dinf_flow_direction",
    "mfd_flow_accumulation",
    "curvature",
    "viewshed",
]

NAN = float("nan")
# D8 codes (ESRI / terra): 1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE; offsets (drow, dcol)
_D8 = [
    (1, (0, 1)),
    (2, (1, 1)),
    (4, (1, 0)),
    (8, (1, -1)),
    (16, (0, -1)),
    (32, (-1, -1)),
    (64, (-1, 0)),
    (128, (-1, 1)),
]
_OFF = dict(_D8)


def _grid(dem):
    G = [[NAN if v is None else float(v) for v in row] for row in dem]
    if not G or any(len(r) != len(G[0]) for r in G):
        raise ValueError("dem must be a rectangular 2-D grid")
    return G


def _ok(v):
    return v == v


def _win(G, i, j):
    """3 x 3 window z1..z9 (row-major from the north-west) or None at edges/NA."""
    nr, nc = len(G), len(G[0])
    if i == 0 or j == 0 or i == nr - 1 or j == nc - 1:
        return None
    w = [G[i + a][j + b] for a in (-1, 0, 1) for b in (-1, 0, 1)]
    return w if all(_ok(v) for v in w) else None


def terrain_indices(dem, *, res: float = 1.0, neighbors: int = 8) -> RichResult:
    r"""Slope, aspect, TPI, TRI, roughness of a DEM, as ``terra::terrain``.

    With the 3 x 3 window ``z1..z9`` (row-major from the north-west) and
    cell size ``res``: slope (radians) ``atan(sqrt(p^2 + q^2))`` from Horn's
    (1981) third-order finite differences ``p = ((z3 + 2 z6 + z9) - (z1 + 2
    z4 + z7)) / (8 res)``, ``q = ((z7 + 2 z8 + z9) - (z1 + 2 z2 + z3)) / (8
    res)`` (``neighbors = 4``: Fleming and Hoffer's ``(z6 - z4)/(2 res)``,
    ``(z8 - z2)/(2 res)``); aspect (radians clockwise from north, downslope
    direction) ``pi/2 - atan2(q, -p)`` folded to ``[0, 2 pi)``;
    ``tpi = z5 - mean(neighbours)``; ``tri`` (Wilson et al. 2007) the mean
    absolute difference to the 8 neighbours; ``tri_riley`` (Riley et al.
    1999) ``sqrt(sum (z_k - z5)^2)``; ``tri_rmsd`` ``sqrt(mean (z_k -
    z5)^2)``; ``roughness`` ``max - min`` of the window.  Edge cells and
    windows with NA are NA.

    References
    ----------
    Horn, B. K. P. (1981). Hill shading and the reflectance map.
    *Proceedings of the IEEE*, 69(1), 14-47.
    Wilson, M. F. J., O'Connell, B., Brown, C., Guinan, J. C. and
    Grehan, A. J. (2007). Multiscale terrain analysis of multibeam
    bathymetry data for habitat mapping on the continental slope. *Marine
    Geodesy*, 30(1-2), 3-35.
    Riley, S. J., DeGloria, S. D. and Elliot, R. (1999). A terrain
    ruggedness index that quantifies topographic heterogeneity.
    *Intermountain Journal of Sciences*, 5(1-4), 23-27.

    Examples
    --------
    >>> t = terrain_indices([[3, 3, 3], [2, 2, 2], [1, 1, 1]])
    >>> round(t.slope[1][1], 6), round(t.aspect[1][1], 6)
    (0.785398, 3.141593)
    """
    G = _grid(dem)
    nr, nc = len(G), len(G[0])
    out = {
        k: [[NAN] * nc for _ in range(nr)]
        for k in ("slope", "aspect", "tpi", "tri", "tri_riley", "tri_rmsd", "roughness")
    }
    for i in range(nr):
        for j in range(nc):
            w = _win(G, i, j)
            if w is None:
                continue
            z1, z2, z3, z4, z5, z6, z7, z8, z9 = w
            if neighbors == 8:
                dx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * res)
                dy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8.0 * res)
            elif neighbors == 4:
                dx = (z6 - z4) / (2.0 * res)
                dy = (z8 - z2) / (2.0 * res)
            else:
                raise ValueError("neighbors must be 4 or 8")
            out["slope"][i][j] = math.atan(math.sqrt(dx * dx + dy * dy))
            a = math.pi / 2.0 - math.atan2(dy, -dx)
            out["aspect"][i][j] = a % (2.0 * math.pi) if (dx != 0 or dy != 0) else NAN
            nb = w[:4] + w[5:]
            out["tpi"][i][j] = z5 - ssum(nb) / 8.0
            out["tri"][i][j] = ssum(abs(v - z5) for v in nb) / 8.0
            out["tri_riley"][i][j] = math.sqrt(ssum((v - z5) ** 2 for v in nb))
            out["tri_rmsd"][i][j] = math.sqrt(ssum((v - z5) ** 2 for v in nb) / 8.0)
            out["roughness"][i][j] = max(w) - min(w)
    return RichResult(payload=out)


def hillshade(slope, aspect, *, angle: float = 45.0, direction: float = 315.0):
    r"""Hillshade ``cos(z) cos(s) + sin(z) sin(s) cos(phi - a)`` with sun zenith ``z = 90 - angle``.

    ``slope`` and ``aspect`` in radians (:func:`terrain_indices`), sun
    ``angle`` (elevation) and ``direction`` (azimuth) in degrees, as
    ``terra::shade`` (Horn 1981); negative values are kept.

    Examples
    --------
    >>> round(hillshade([[0.0]], [[0.0]])[0][0], 6)
    0.707107
    """
    zen = math.radians(90.0 - angle)
    az = math.radians(direction)
    return [
        [
            math.cos(zen) * math.cos(s) + math.sin(zen) * math.sin(s) * math.cos(az - a)
            if _ok(s) and _ok(a)
            else (math.cos(zen) if _ok(s) and s == 0 else NAN)
            for s, a in zip(rs, ra)
        ]
        for rs, ra in zip(slope, aspect)
    ]


def d8_flow_direction(dem, *, res: float = 1.0):
    r"""D8 flow direction (O'Callaghan and Mark 1984), coded as ``terra::terrain(v = "flowdir")``.

    Each cell drains to the neighbour of steepest descent ``(z - z_k)/d_k``
    (``d_k = res`` or ``res sqrt 2``); codes 1 E, 2 SE, 4 S, 8 SW, 16 W, 32
    NW, 64 N, 128 NE; 0 where no neighbour is lower (pits and flats); ties go
    to the first code in that order.  Cells on the grid edge drain off the
    grid in the steepest outward direction when no interior neighbour is
    lower.

    References
    ----------
    O'Callaghan, J. F. and Mark, D. M. (1984). The extraction of drainage
    networks from digital elevation data. *Computer Vision, Graphics, and
    Image Processing*, 28(3), 323-344.

    Examples
    --------
    >>> d8_flow_direction([[3, 3, 3], [2, 2, 2], [1, 1, 1]])[1][1]
    4
    """
    G = _grid(dem)
    nr, nc = len(G), len(G[0])
    out = [[0] * nc for _ in range(nr)]
    for i in range(nr):
        for j in range(nc):
            z = G[i][j]
            if not _ok(z):
                out[i][j] = 0
                continue
            best, code = 0.0, 0
            for c, (a, b) in _D8:
                x, y = i + a, j + b
                if not (0 <= x < nr and 0 <= y < nc) or not _ok(G[x][y]):
                    continue
                s = (z - G[x][y]) / (res * (math.sqrt(2.0) if a and b else 1.0))
                if s > best:
                    best, code = s, c
            out[i][j] = code
    return out


def _receivers(fd):
    nr, nc = len(fd), len(fd[0])
    rec = {}
    for i in range(nr):
        for j in range(nc):
            c = fd[i][j]
            if c in _OFF:
                a, b = _OFF[c]
                x, y = i + a, j + b
                if 0 <= x < nr and 0 <= y < nc:
                    rec[(i, j)] = (x, y)
    return rec


def flow_accumulation(flowdir, weight=None):
    r"""Number (or weighted sum) of cells draining through each cell, itself included.

    Follows the D8 codes of :func:`d8_flow_direction` in topological order
    (upstream first), as ``terra::flowAccumulation``.

    Examples
    --------
    >>> flow_accumulation([[4, 4], [1, 0]])
    [[1.0, 1.0], [2.0, 4.0]]
    """
    nr, nc = len(flowdir), len(flowdir[0])
    rec = _receivers(flowdir)
    acc = [[float(weight[i][j]) if weight is not None else 1.0 for j in range(nc)] for i in range(nr)]
    indeg = {}
    for d in rec.values():
        indeg[d] = indeg.get(d, 0) + 1
    stack = [(i, j) for i in range(nr) for j in range(nc) if indeg.get((i, j), 0) == 0]
    while stack:
        c = stack.pop()
        d = rec.get(c)
        if d is None:
            continue
        acc[d[0]][d[1]] += acc[c[0]][c[1]]
        indeg[d] -= 1
        if indeg[d] == 0:
            stack.append(d)
    return acc


def watershed(flowdir, outlet):
    r"""Cells draining (by D8) to the outlet cell ``(row, col)``: 1 inside the watershed, 0 outside.

    Examples
    --------
    >>> watershed([[4, 4], [1, 0]], (1, 1))
    [[1, 1], [1, 1]]
    """
    nr, nc = len(flowdir), len(flowdir[0])
    rec = _receivers(flowdir)
    up = {}
    for s, d in rec.items():
        up.setdefault(d, []).append(s)
    out = [[0] * nc for _ in range(nr)]
    stack = [tuple(outlet)]
    while stack:
        i, j = stack.pop()
        if out[i][j]:
            continue
        out[i][j] = 1
        stack.extend(up.get((i, j), []))
    return out


def fill_sinks(dem, *, epsilon: float = 0.0):
    r"""Fill depressions by the priority-flood algorithm (Barnes, Lehman and Mulla 2014).

    Cells are raised to the lowest spill elevation reachable from the grid
    edge (or an NA cell); with ``epsilon > 0`` each filled cell is raised
    ``epsilon`` above the cell it was reached from, so every cell has a
    downslope path (the "+epsilon" variant).  NA cells are kept.

    References
    ----------
    Barnes, R., Lehman, C. and Mulla, D. (2014). Priority-flood: an optimal
    depression-filling and watershed-labeling algorithm for digital
    elevation models. *Computers and Geosciences*, 62, 117-127.

    Examples
    --------
    >>> fill_sinks([[5, 5, 5], [5, 1, 5], [5, 5, 5]])[1][1]
    5.0
    """
    G = _grid(dem)
    nr, nc = len(G), len(G[0])
    Z = [r[:] for r in G]
    done = [[False] * nc for _ in range(nr)]
    pq = []
    for i in range(nr):
        for j in range(nc):
            if not _ok(Z[i][j]):
                done[i][j] = True
                continue
            edge = (
                i in (0, nr - 1)
                or j in (0, nc - 1)
                or any(not _ok(G[i + a][j + b]) for _, (a, b) in _D8 if 0 <= i + a < nr and 0 <= j + b < nc)
            )
            if edge:
                heapq.heappush(pq, (Z[i][j], i, j))
                done[i][j] = True
    while pq:
        z, i, j = heapq.heappop(pq)
        for _, (a, b) in _D8:
            x, y = i + a, j + b
            if 0 <= x < nr and 0 <= y < nc and not done[x][y]:
                done[x][y] = True
                Z[x][y] = max(Z[x][y], z + epsilon) if epsilon > 0 else max(Z[x][y], z)
                heapq.heappush(pq, (Z[x][y], x, y))
    return Z


def stream_order(flowdir, acc, threshold: float, *, method: str = "strahler"):
    r"""Stream network order: cells with accumulation ``>= threshold`` form the network.

    ``strahler`` (Strahler 1957): sources are 1; a cell takes the maximum
    order of its network donors, plus one when two or more donors share that
    maximum.  ``shreve`` (Shreve 1966): sources are 1 and the magnitude is the
    sum over donors.  Non-network cells are 0.

    References
    ----------
    Strahler, A. N. (1957). Quantitative analysis of watershed
    geomorphology. *Transactions, American Geophysical Union*, 38(6),
    913-920.
    Shreve, R. L. (1966). Statistical law of stream numbers. *Journal of
    Geology*, 74(1), 17-37.

    Examples
    --------
    >>> fd = [[2, 4, 8], [0, 4, 0], [0, 0, 0]]
    >>> stream_order(fd, flow_accumulation(fd), 1)[1][1]
    2
    """
    if method not in ("strahler", "shreve"):
        raise ValueError("method must be strahler or shreve")
    nr, nc = len(flowdir), len(flowdir[0])
    net = [[acc[i][j] >= threshold for j in range(nc)] for i in range(nr)]
    rec = {c: d for c, d in _receivers(flowdir).items() if net[c[0]][c[1]] and net[d[0]][d[1]]}
    donors = {}
    for s, d in rec.items():
        donors.setdefault(d, []).append(s)
    order = [[0] * nc for _ in range(nr)]
    indeg = {d: len(v) for d, v in donors.items()}
    stack = [(i, j) for i in range(nr) for j in range(nc) if net[i][j] and indeg.get((i, j), 0) == 0]
    while stack:
        c = stack.pop()
        ds = donors.get(c, [])
        if not ds:
            order[c[0]][c[1]] = 1
        elif method == "shreve":
            order[c[0]][c[1]] = sum(order[a][b] for a, b in ds)
        else:
            m = max(order[a][b] for a, b in ds)
            order[c[0]][c[1]] = m + 1 if sum(1 for a, b in ds if order[a][b] == m) >= 2 else m
        d = rec.get(c)
        if d is not None:
            indeg[d] -= 1
            if indeg[d] == 0:
                stack.append(d)
    return order


def topographic_wetness_index(acc, slope, *, res: float = 1.0, min_slope: float = 1e-6):
    r"""Topographic wetness index ``ln(a / tan(beta))`` (Beven and Kirkby 1979).

    ``a = acc res`` is the specific catchment area (upslope area per unit
    contour width: accumulated cells times ``res^2`` over ``res``) and
    ``beta`` the slope in radians, floored at ``min_slope``.

    References
    ----------
    Beven, K. J. and Kirkby, M. J. (1979). A physically based, variable
    contributing area model of basin hydrology. *Hydrological Sciences
    Bulletin*, 24(1), 43-69.

    Examples
    --------
    >>> round(topographic_wetness_index([[4.0]], [[math.atan(0.5)]], res=2.0)[0][0], 6)
    2.772589
    """
    return [
        [math.log(a * res / math.tan(max(s, min_slope))) if _ok(s) else NAN for a, s in zip(ra, rs)]
        for ra, rs in zip(acc, slope)
    ]


def stream_power_index(acc, slope, *, res: float = 1.0):
    r"""Stream power index ``a tan(beta)`` with ``a = acc res`` (Moore, Grayson and Ladson 1991).

    References
    ----------
    Moore, I. D., Grayson, R. B. and Ladson, A. R. (1991). Digital terrain
    modelling: a review of hydrological, geomorphological, and biological
    applications. *Hydrological Processes*, 5(1), 3-30.

    Examples
    --------
    >>> round(stream_power_index([[4.0]], [[math.atan(0.5)]], res=2.0)[0][0], 6)
    4.0
    """
    return [[a * res * math.tan(s) if _ok(s) else NAN for a, s in zip(ra, rs)] for ra, rs in zip(acc, slope)]


def dinf_flow_direction(dem, *, res: float = 1.0):
    r"""D-infinity flow direction (Tarboton 1997), radians counter-clockwise from east.

    Over the eight triangular facets around each cell the steepest downward
    slope is found; its angle is ``r = atan(s2/s1)`` clamped to the facet
    (slope ``sqrt(s1^2 + s2^2)``, or the facet edge slope when clamped), and
    the direction is ``a_c r + a_f pi/2``.  ``nan`` where no facet
    descends; interior cells only.

    References
    ----------
    Tarboton, D. G. (1997). A new method for the determination of flow
    directions and upslope areas in grid digital elevation models. *Water
    Resources Research*, 33(2), 309-319.

    Examples
    --------
    >>> round(dinf_flow_direction([[3, 3, 3], [2, 2, 2], [1, 1, 1]])[1][1], 6)
    4.712389
    """
    G = _grid(dem)
    nr, nc = len(G), len(G[0])
    # facets: (e1 offset, e2 offset, a_c, a_f) with e1 cardinal, e2 diagonal (Tarboton 1997, table 1)
    fac = [
        ((0, 1), (-1, 1), 0, 1),
        ((-1, 0), (-1, 1), 1, -1),
        ((-1, 0), (-1, -1), 1, 1),
        ((0, -1), (-1, -1), 2, -1),
        ((0, -1), (1, -1), 2, 1),
        ((1, 0), (1, -1), 3, -1),
        ((1, 0), (1, 1), 3, 1),
        ((0, 1), (1, 1), 4, -1),
    ]
    out = [[NAN] * nc for _ in range(nr)]
    for i in range(1, nr - 1):
        for j in range(1, nc - 1):
            w = _win(G, i, j)
            if w is None:
                continue
            e0 = G[i][j]
            best, ang = 0.0, NAN
            for (a1, b1), (a2, b2), ac, af in fac:
                e1, e2 = G[i + a1][j + b1], G[i + a2][j + b2]
                s1 = (e0 - e1) / res
                s2 = (e1 - e2) / res
                r = math.atan2(s2, s1)
                s = math.hypot(s1, s2)
                if r < 0:
                    r, s = 0.0, s1
                elif r > math.pi / 4:
                    r, s = math.pi / 4, (e0 - e2) / (res * math.sqrt(2.0))
                if s > best:
                    best, ang = s, ac * math.pi / 2 + af * r
            out[i][j] = ang % (2 * math.pi) if ang == ang else NAN
    return out


def mfd_flow_accumulation(dem, *, res: float = 1.0, p: float = 1.1):
    r"""Multiple flow direction accumulation (Freeman 1991; Quinn et al. 1991 with ``p = 1``).

    Each cell passes its accumulated area to every lower neighbour in
    proportion to ``(tan beta_k)^p L_k`` with ``tan beta_k = (z - z_k)/d_k``
    and contour length ``L_k = 0.5 res`` (cardinal) or ``0.354 res``
    (diagonal) for Quinn et al.; Freeman's form uses ``L_k = 1`` and ``p =
    1.1``.  Here ``L_k = 1`` (Freeman) for any ``p``; cells are processed
    from high to low.

    References
    ----------
    Freeman, T. G. (1991). Calculating catchment area with divergent flow
    based on a regular grid. *Computers and Geosciences*, 17(3), 413-422.

    Examples
    --------
    >>> [round(v, 6) for v in mfd_flow_accumulation([[2, 1], [1, 0]])[1]]
    [1.288676, 4.0]
    """
    G = _grid(dem)
    nr, nc = len(G), len(G[0])
    acc = [[1.0 if _ok(G[i][j]) else NAN for j in range(nc)] for i in range(nr)]
    cells = sorted(((G[i][j], i, j) for i in range(nr) for j in range(nc) if _ok(G[i][j])), reverse=True)
    for z, i, j in cells:
        wts = []
        for _, (a, b) in _D8:
            x, y = i + a, j + b
            if 0 <= x < nr and 0 <= y < nc and _ok(G[x][y]) and G[x][y] < z:
                wts.append(((x, y), ((z - G[x][y]) / (res * (math.sqrt(2.0) if a and b else 1.0))) ** p))
        tot = ssum(w for _, w in wts)
        for (x, y), w in wts:
            acc[x][y] += acc[i][j] * w / tot
    return acc


def curvature(dem, *, res: float = 1.0) -> RichResult:
    r"""Profile, plan and total curvature (Zevenbergen and Thorne 1987).

    With ``D = ((z4 + z6)/2 - z5)/L^2``, ``E = ((z2 + z8)/2 - z5)/L^2``,
    ``F = (z3 - z1 + z7 - z9)/(4 L^2)``, ``G = (z6 - z4)/(2L)``, ``H = (z2 -
    z8)/(2L)``: ``profile = -2 (D G^2 + E H^2 + F G H)/(G^2 + H^2)``,
    ``plan = 2 (D H^2 + E G^2 - F G H)/(G^2 + H^2)``, ``total = -2 (D + E)``
    (units 1/length; profile and plan ``nan`` on flat cells).

    References
    ----------
    Zevenbergen, L. W. and Thorne, C. R. (1987). Quantitative analysis of
    land surface topography. *Earth Surface Processes and Landforms*, 12(1),
    47-56.

    Examples
    --------
    >>> c = curvature([[2, 1, 2], [1, 0, 1], [2, 1, 2]])
    >>> round(c.total[1][1], 6)
    -4.0
    """
    G = _grid(dem)
    nr, nc = len(G), len(G[0])
    out = {k: [[NAN] * nc for _ in range(nr)] for k in ("profile", "plan", "total")}
    L = float(res)
    for i in range(nr):
        for j in range(nc):
            w = _win(G, i, j)
            if w is None:
                continue
            z1, z2, z3, z4, z5, z6, z7, z8, z9 = w
            D = ((z4 + z6) / 2 - z5) / L**2
            E = ((z2 + z8) / 2 - z5) / L**2
            F = (z3 - z1 + z7 - z9) / (4 * L**2)
            Gx = (z6 - z4) / (2 * L)
            H = (z2 - z8) / (2 * L)
            out["total"][i][j] = -2.0 * (D + E)
            g2 = Gx * Gx + H * H
            if g2 > 0:
                out["profile"][i][j] = -2.0 * (D * Gx * Gx + E * H * H + F * Gx * H) / g2
                out["plan"][i][j] = 2.0 * (D * H * H + E * Gx * Gx - F * Gx * H) / g2
    return RichResult(payload=out)


def viewshed(dem, observer, *, res: float = 1.0, observer_height: float = 1.7, target_height: float = 0.0):
    r"""Line-of-sight viewshed from the cell ``observer = (row, col)``.

    A target cell is visible when no intermediate cell sampled along the
    straight line (at every cell crossed, bilinear-free nearest-cell
    elevations at unit steps of the longer axis) rises above the sight line
    from the observer eye ``z_o + observer_height`` to ``z_t +
    target_height`` (Franklin and Ray 1994, R3 algorithm; no earth
    curvature).

    References
    ----------
    Franklin, W. R. and Ray, C. K. (1994). Higher isn't necessarily better:
    visibility algorithms and experiments. *Advances in GIS Research: Sixth
    International Symposium on Spatial Data Handling*, 751-770.

    Examples
    --------
    >>> viewshed([[0, 0, 0, 0], [0, 0, 9, 0]], (0, 0))[1][3]
    0
    """
    G = _grid(dem)
    nr, nc = len(G), len(G[0])
    oi, oj = observer
    eye = G[oi][oj] + observer_height
    out = [[0] * nc for _ in range(nr)]
    for ti in range(nr):
        for tj in range(nc):
            if not _ok(G[ti][tj]):
                continue
            n = max(abs(ti - oi), abs(tj - oj))
            if n == 0:
                out[ti][tj] = 1
                continue
            tz = G[ti][tj] + target_height
            vis = 1
            for k in range(1, n):
                t = k / n
                ci, cj = round(oi + t * (ti - oi)), round(oj + t * (tj - oj))
                z = G[ci][cj]
                if _ok(z) and z > eye + t * (tz - eye):
                    vis = 0
                    break
            out[ti][tj] = vis
    return out


def cheatsheet() -> str:
    return "terrain_indices / d8_flow_direction / flow_accumulation / fill_sinks / stream_order -> DEM analysis."

# alias kept from the retired placeholder of the same name
gradient_spatial = terrain_indices
