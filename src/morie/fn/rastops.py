# morie.fn -- function file (rootcoder007/morie)
"""Raster operations: focal statistics and convolution filters (median, range, variance, Gaussian, Laplacian,
Prewitt, sharpen, binomial, Gabor, entropy), Canny edges, grey morphology, aggregation, disaggregation,
bilinear resampling, zonal statistics, masking, Euclidean distance and accumulated cost surfaces."""

from __future__ import annotations

import heapq
import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "focal_statistics",
    "focal_filter",
    "filter_kernel",
    "canny_edges",
    "grey_morphology",
    "raster_aggregate",
    "raster_disaggregate",
    "raster_resample",
    "zonal_statistics",
    "raster_mask",
    "distance_transform",
    "cost_distance",
]

_NAN = float("nan")


def _grid(g):
    return [[float(v) if v is not None else _NAN for v in row] for row in np.asarray(g, dtype=float).tolist()]


def _isnan(v):
    return v != v


def _window(G, i, j, h, w):
    nr, nc = len(G), len(G[0])
    out = []
    for a in range(i - h, i + h + 1):
        for b in range(j - w, j + w + 1):
            out.append(G[a][b] if 0 <= a < nr and 0 <= b < nc else _NAN)
    return out


def _stat(v, fun):
    n = len(v)
    if fun == "mean":
        return ssum(v) / n
    if fun == "sum":
        return ssum(v)
    if fun == "min":
        return min(v)
    if fun == "max":
        return max(v)
    if fun == "range":
        return max(v) - min(v)
    if fun == "median":
        s = sorted(v)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    if fun in ("var", "sd"):
        if n < 2:
            return _NAN
        m = ssum(v) / n
        var = ssum((x - m) ** 2 for x in v) / (n - 1)
        return var if fun == "var" else math.sqrt(var)
    if fun == "entropy":
        cnt = {}
        for x in v:
            cnt[x] = cnt.get(x, 0) + 1
        return -ssum(c / n * math.log(c / n) for c in cnt.values())
    raise ValueError("unknown focal function")


def focal_statistics(grid, size: int = 3, fun: str = "mean", *, na_rm: bool = False):
    r"""Moving-window (focal) statistics over a ``size`` x ``size`` window, as ``terra::focal(x, w, fun)``.

    ``fun`` is ``mean``, ``sum``, ``min``, ``max``, ``range``, ``median``,
    ``var`` or ``sd`` (sample, divisor ``n - 1``) or ``entropy`` (Shannon
    entropy, natural log, of the value frequencies in the window). Cells
    outside the grid are ``NA``; with ``na_rm=False`` any ``NA`` in the
    window gives ``NA``, otherwise ``NA`` values are dropped (``NA`` if none
    remain).

    Examples
    --------
    >>> focal_statistics([[1, 2, 3], [4, 5, 6], [7, 8, 9]], fun="median", na_rm=True)[1]
    [4.5, 5.0, 5.5]
    """
    if size % 2 != 1:
        raise ValueError("size must be odd")
    G = _grid(grid)
    h = size // 2
    out = []
    for i in range(len(G)):
        row = []
        for j in range(len(G[0])):
            v = _window(G, i, j, h, h)
            if any(_isnan(x) for x in v):
                if not na_rm:
                    row.append(_NAN)
                    continue
                v = [x for x in v if not _isnan(x)]
                if not v:
                    row.append(_NAN)
                    continue
            row.append(_stat(v, fun))
        out.append(row)
    return out


def filter_kernel(
    name: str,
    *,
    size: int = 3,
    sigma: float = 1.0,
    theta: float = 0.0,
    wavelength: float = 4.0,
    gamma: float = 0.5,
    psi: float = 0.0,
):
    r"""Standard filter kernels (rows north to south, columns west to east).

    ``mean`` (box ``1/size^2``), ``binomial`` (``[1, 2, 1]`` outer product
    ``/16`` for size 3; binomial coefficients in general), ``gaussian``
    (``exp(-(x^2 + y^2)/(2 sigma^2))`` normalised to sum 1), ``laplacian``
    (``[[0, 1, 0], [1, -4, 1], [0, 1, 0]]``), ``laplacian8`` (``-8`` centre),
    ``sharpen`` (``[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]``), ``prewitt_x``
    (``[[-1, 0, 1]] * 3``), ``prewitt_y`` (``[-1, -1, -1]``, ``0``, ``[1, 1,
    1]`` by row), ``sobel_x``, ``sobel_y`` and ``gabor``: ``exp(-(x'^2 +
    gamma^2 y'^2)/(2 sigma^2)) cos(2 pi x'/wavelength + psi)`` with ``x' = x
    cos theta + y sin theta``, ``y' = -x sin theta + y cos theta`` (Daugman
    1985), ``y`` increasing southwards.

    Examples
    --------
    >>> filter_kernel("binomial")
    [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]
    """
    h = size // 2
    if name == "mean":
        return [[1 / size**2] * size for _ in range(size)]
    if name == "binomial":
        b = [math.comb(size - 1, k) for k in range(size)]
        s = sum(b) ** 2
        return [[x * y / s for y in b] for x in b]
    if name == "gaussian":
        K = [[math.exp(-((a - h) ** 2 + (b - h) ** 2) / (2 * sigma * sigma)) for b in range(size)] for a in range(size)]
        s = ssum(ssum(r) for r in K)
        return [[v / s for v in r] for r in K]
    if name == "gabor":
        K = []
        for a in range(size):
            row = []
            for b in range(size):
                x, y = b - h, a - h
                xp = x * math.cos(theta) + y * math.sin(theta)
                yp = -x * math.sin(theta) + y * math.cos(theta)
                row.append(
                    math.exp(-(xp * xp + gamma * gamma * yp * yp) / (2 * sigma * sigma))
                    * math.cos(2 * math.pi * xp / wavelength + psi)
                )
            K.append(row)
        return K
    fixed = {
        "laplacian": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        "laplacian8": [[1, 1, 1], [1, -8, 1], [1, 1, 1]],
        "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "prewitt_x": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        "prewitt_y": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        "sobel_x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "sobel_y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    }
    if name not in fixed:
        raise ValueError("unknown kernel")
    return [[float(v) for v in r] for r in fixed[name]]


def focal_filter(grid, kernel, *, na_rm: bool = False):
    r"""Weighted moving-window sum ``sum_k w_k z_k`` with the kernel laid over the window as given (correlation).

    Equivalent to ``terra::focal(x, w = kernel, fun = "sum")``: edge cells and
    windows holding ``NA`` are ``NA`` unless ``na_rm`` (then ``NA`` cells
    contribute nothing). Kernels come from :func:`filter_kernel`; a gradient
    magnitude is ``sqrt(gx^2 + gy^2)`` of the ``*_x`` and ``*_y`` outputs.

    Examples
    --------
    >>> focal_filter([[1, 2, 3], [4, 5, 6], [7, 8, 9]], filter_kernel("prewitt_x"))[1][1]
    6.0
    """
    G = _grid(grid)
    K = [[float(v) for v in r] for r in kernel]
    kh, kw = len(K) // 2, len(K[0]) // 2
    wflat = [v for r in K for v in r]
    out = []
    for i in range(len(G)):
        row = []
        for j in range(len(G[0])):
            v = _window(G, i, j, kh, kw)
            if any(_isnan(x) for x in v) and not na_rm:
                row.append(_NAN)
                continue
            pairs = [(w, x) for w, x in zip(wflat, v) if not _isnan(x)]
            row.append(ssum(w * x for w, x in pairs) if pairs else _NAN)
        out.append(row)
    return out


def canny_edges(grid, *, sigma: float = 1.0, low: float = 0.1, high: float = 0.2, size: int = 5):
    r"""Canny (1986) edge detector: Gaussian smoothing, Sobel gradients, non-maximum suppression, hysteresis.

    The grid is smoothed with a ``size`` x ``size`` Gaussian kernel
    renormalised over the in-grid, non-``NA`` cells, gradients use the Sobel kernels, the magnitude is suppressed
    unless it is a maximum along the gradient direction quantised to 0, 45,
    90 or 135 degrees (ties within a relative ``1e-9`` kept;
    magnitudes below ``1e-12`` of the largest smoothed value count as 0), and cells above ``high`` (a fraction of the
    maximum magnitude) seed edges that grow through 8-connected cells above
    ``low``. Returns a 0/1 grid (border cells 0).

    References
    ----------
    Canny, J. (1986). A computational approach to edge detection. *IEEE
    Transactions on Pattern Analysis and Machine Intelligence*, 8(6), 679-698.

    Examples
    --------
    >>> E = canny_edges([[0, 0, 0, 9, 9, 9]] * 6, sigma=0.5, size=3)
    >>> E[2]
    [0, 0, 1, 1, 0, 0]
    """
    K = filter_kernel("gaussian", size=size, sigma=sigma)
    G0 = _grid(grid)
    num = focal_filter(G0, K, na_rm=True)
    den = focal_filter([[_NAN if _isnan(v) else 1.0 for v in r] for r in G0], K, na_rm=True)
    S = [[a / b if not _isnan(a) and b > 0 else _NAN for a, b in zip(ra, rb)] for ra, rb in zip(num, den)]
    gx = focal_filter(S, filter_kernel("sobel_x"))
    gy = focal_filter(S, filter_kernel("sobel_y"))
    nr, nc = len(S), len(S[0])
    M = [[math.hypot(gx[i][j], gy[i][j]) if not _isnan(gx[i][j]) else 0.0 for j in range(nc)] for i in range(nr)]
    scale = max((abs(v) for r in S for v in r if not _isnan(v)), default=0.0)
    M = [[v if v > 1e-12 * scale else 0.0 for v in r] for r in M]
    mx = max(max(r) for r in M) or 1.0
    N = [[0.0] * nc for _ in range(nr)]
    for i in range(1, nr - 1):
        for j in range(1, nc - 1):
            if M[i][j] == 0:
                continue
            ang = math.degrees(math.atan2(gy[i][j], gx[i][j])) % 180
            if ang < 22.5 or ang >= 157.5:
                a, b = M[i][j - 1], M[i][j + 1]
            elif ang < 67.5:
                a, b = M[i + 1][j + 1], M[i - 1][j - 1]
            elif ang < 112.5:
                a, b = M[i - 1][j], M[i + 1][j]
            else:
                a, b = M[i + 1][j - 1], M[i - 1][j + 1]
            tol = 1e-9 * M[i][j]
            if M[i][j] >= a - tol and M[i][j] >= b - tol:
                N[i][j] = M[i][j] / mx
    E = [[0] * nc for _ in range(nr)]
    stack = [(i, j) for i in range(nr) for j in range(nc) if N[i][j] >= high]
    for i, j in stack:
        E[i][j] = 1
    while stack:
        i, j = stack.pop()
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                u, v = i + a, j + b
                if 0 <= u < nr and 0 <= v < nc and not E[u][v] and N[u][v] >= low:
                    E[u][v] = 1
                    stack.append((u, v))
    return E


def grey_morphology(grid, operation: str, size: int = 3):
    r"""Grey-scale morphology with a flat square structuring element (Serra 1982).

    ``erosion`` = focal minimum, ``dilation`` = focal maximum, ``opening`` =
    dilation of the erosion, ``closing`` = erosion of the dilation, over the
    in-grid cells of the window (``NA`` ignored). On a 0/1 grid these are the
    binary operations.

    References
    ----------
    Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.

    Examples
    --------
    >>> grey_morphology([[0, 0, 0], [0, 1, 0], [0, 0, 0]], "dilation")[0]
    [1.0, 1.0, 1.0]
    """
    ops = {"erosion": ["min"], "dilation": ["max"], "opening": ["min", "max"], "closing": ["max", "min"]}
    if operation not in ops:
        raise ValueError("operation must be erosion, dilation, opening or closing")
    G = _grid(grid)
    for f in ops[operation]:
        G = focal_statistics(G, size, f, na_rm=True)
    return G


def raster_aggregate(grid, fact: int, fun: str = "mean", *, na_rm: bool = False):
    r"""Aggregate ``fact`` x ``fact`` blocks into coarser cells, as ``terra::aggregate`` (partial edge blocks kept).

    Examples
    --------
    >>> raster_aggregate([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2)
    [[3.0, 4.5], [7.5, 9.0]]
    """
    G = _grid(grid)
    nr, nc = len(G), len(G[0])
    out = []
    for i in range(0, nr, fact):
        row = []
        for j in range(0, nc, fact):
            v = [G[a][b] for a in range(i, min(i + fact, nr)) for b in range(j, min(j + fact, nc))]
            if any(_isnan(x) for x in v):
                if not na_rm:
                    row.append(_NAN)
                    continue
                v = [x for x in v if not _isnan(x)]
            row.append(_stat(v, fun) if v else _NAN)
        out.append(row)
    return out


def raster_disaggregate(grid, fact: int):
    r"""Split each cell into ``fact`` x ``fact`` cells carrying its value (``terra::disagg``, method ``near``).

    Examples
    --------
    >>> raster_disaggregate([[1, 2]], 2)
    [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]
    """
    G = _grid(grid)
    return [[v for v in row for _ in range(fact)] for row in G for _ in range(fact)]


def raster_resample(grid, extent, new_x, new_y, *, method: str = "bilinear"):
    r"""Values of a raster at arbitrary points (e.g. the cell centres of a new grid).

    ``extent = (xmin, xmax, ymin, ymax)`` of the source grid (row 0 at
    ``ymax``); cell centres are at ``xmin + (j + 1/2) dx`` and ``ymax - (i +
    1/2) dy``. ``bilinear`` interpolates between the four surrounding centres
    (``NA`` outside the hull of the centres or next to ``NA``); ``near`` takes
    the containing cell. Returns ``len(new_y)`` rows by ``len(new_x)`` columns.

    Examples
    --------
    >>> raster_resample([[1, 2], [3, 4]], (0, 2, 0, 2), [1.0], [1.0])
    [[2.5]]
    """
    if method not in ("bilinear", "near"):
        raise ValueError("method must be bilinear or near")
    G = _grid(grid)
    nr, nc = len(G), len(G[0])
    xmin, xmax, ymin, ymax = (float(v) for v in extent)
    dx, dy = (xmax - xmin) / nc, (ymax - ymin) / nr
    out = []
    for y in new_y:
        row = []
        for x in new_x:
            if method == "near":
                j, i = math.floor((x - xmin) / dx), math.floor((ymax - y) / dy)
                row.append(G[i][j] if 0 <= i < nr and 0 <= j < nc else _NAN)
                continue
            fx, fy = (x - xmin) / dx - 0.5, (ymax - y) / dy - 0.5
            j0, i0 = math.floor(fx), math.floor(fy)
            if fx == nc - 1:
                j0 -= 1
            if fy == nr - 1:
                i0 -= 1
            if not (0 <= j0 < nc - 1 and 0 <= i0 < nr - 1) or fx < 0 or fy < 0:
                row.append(_NAN)
                continue
            tx, ty = fx - j0, fy - i0
            z = [G[i0][j0], G[i0][j0 + 1], G[i0 + 1][j0], G[i0 + 1][j0 + 1]]
            if any(_isnan(v) for v in z):
                row.append(_NAN)
                continue
            row.append((1 - ty) * ((1 - tx) * z[0] + tx * z[1]) + ty * ((1 - tx) * z[2] + tx * z[3]))
        out.append(row)
    return out


def zonal_statistics(grid, zones, fun: str = "mean") -> RichResult:
    r"""Statistic of the grid values per zone label (``NA`` values and ``NA`` zones skipped), as ``terra::zonal``.

    Examples
    --------
    >>> r = zonal_statistics([[1, 2], [3, 4]], [[1, 1, ], [2, 2]])
    >>> r.zone, r.value
    ([1.0, 2.0], [1.5, 3.5])
    """
    G, Z = _grid(grid), _grid(zones)
    vals = {}
    for gr, zr in zip(G, Z):
        for v, z in zip(gr, zr):
            if not _isnan(v) and not _isnan(z):
                vals.setdefault(z, []).append(v)
    keys = sorted(vals)
    return RichResult(
        payload={"zone": keys, "value": [_stat(vals[k], fun) for k in keys], "count": [len(vals[k]) for k in keys]}
    )


def raster_mask(grid, mask, *, maskvalue=None, inverse: bool = False):
    r"""Set cells to ``NA`` where ``mask`` is ``NA`` (or equals ``maskvalue``), as ``terra::mask``; ``inverse`` flips it.

    Examples
    --------
    >>> raster_mask([[1, 2], [3, 4]], [[1, None], [1, 1]])
    [[1.0, nan], [3.0, 4.0]]
    """
    G, M = _grid(grid), _grid(mask)
    out = []
    for gr, mr in zip(G, M):
        row = []
        for v, m in zip(gr, mr):
            hit = _isnan(m) if maskvalue is None else (m == maskvalue)
            row.append(_NAN if hit != inverse else v)
        out.append(row)
    return out


def distance_transform(grid, *, res: float = 1.0):
    r"""Euclidean distance from every cell centre to the nearest non-``NA`` cell (0 on those), as ``terra::distance``.

    Exact brute force over the target cells (planar coordinates, square
    cells of side ``res``).

    Examples
    --------
    >>> distance_transform([[1, None, None], [None, None, None]])
    [[0.0, 1.0, 2.0], [1.0, 1.4142135623730951, 2.23606797749979]]
    """
    G = _grid(grid)
    T = [(i, j) for i, r in enumerate(G) for j, v in enumerate(r) if not _isnan(v)]
    if not T:
        raise ValueError("no target cells")
    return [
        [0.0 if not _isnan(v) else res * min(math.hypot(i - a, j - b) for a, b in T) for j, v in enumerate(r)]
        for i, r in enumerate(G)
    ]


def cost_distance(cost, sources, *, res: float = 1.0):
    r"""Accumulated least-cost distance from source cells over a friction surface (8 neighbours).

    A step between neighbouring cells costs the mean of their friction values
    times the step length (``res`` or ``res sqrt(2)``; the ArcGIS Cost Distance
    and ``gdistance`` convention);
    ``NA`` cells are barriers. ``sources`` are ``(row, col)`` pairs. Dijkstra
    (1959) over the cell graph.

    Examples
    --------
    >>> cost_distance([[1, 1, 1], [1, 9, 1], [1, 1, 1]], [(0, 0)])[2][2]
    3.414213562373095
    """
    C = _grid(cost)
    nr, nc = len(C), len(C[0])
    D = [[math.inf] * nc for _ in range(nr)]
    h = []
    for i, j in sources:
        D[i][j] = 0.0
        heapq.heappush(h, (0.0, i, j))
    while h:
        d, i, j = heapq.heappop(h)
        if d > D[i][j]:
            continue
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                u, v = i + a, j + b
                if (a or b) and 0 <= u < nr and 0 <= v < nc and not _isnan(C[u][v]):
                    step = res * (math.sqrt(2) if a and b else 1.0)
                    nd = d + step * (C[i][j] + C[u][v]) / 2
                    if nd < D[u][v]:
                        D[u][v] = nd
                        heapq.heappush(h, (nd, u, v))
    return [[_NAN if _isnan(C[i][j]) else D[i][j] for j in range(nc)] for i in range(nr)]


def cheatsheet() -> str:
    return "focal_statistics / focal_filter / canny_edges / raster_aggregate / cost_distance -> raster operations."
