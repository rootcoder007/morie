# morie.fn -- function file (rootcoder007/morie)
"""Forest inventory and LiDAR canopy analysis: basal area, stand density index (Reineke and
summation), allometric above-ground biomass (Chave et al. 2014) and carbon stock, point-cloud
rasterisation to surface and terrain models, canopy height models, canopy gap delineation,
variable-window tree-top detection (Popescu and Wynne 2004), region-growing crown segmentation
(Dalponte and Coomes 2016), and the sampling estimators of fixed-area plots, stratified
inventories, adaptive cluster sampling (Thompson 1990) and line-intersect sampling of downed
wood (Van Wagner 1968).

Rasters are lists of rows; row 0 is the northern (largest y) edge and column 0 the western
edge; ``nan`` marks empty cells."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "basal_area",
    "stand_density_index",
    "tree_biomass",
    "carbon_stock",
    "rasterize_points",
    "canopy_height_model",
    "canopy_gaps",
    "tree_tops",
    "crown_segmentation",
    "plot_estimate",
    "stratified_estimate",
    "adaptive_cluster_estimate",
    "line_intersect_volume",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _grid(g):
    return [[float(v) for v in r] for r in np.asarray(g, dtype=float).tolist()]


def _rep(v, n):
    return [float(v)] * n if isinstance(v, (int, float)) else _vec(v)


def basal_area(dbh, expansion=1.0):
    r"""Tree and stand basal area from diameters at breast height (cm).

    ``g_i = pi d_i^2 / 40000`` m^2 and ``BA = sum g_i e_i`` with expansion
    factors ``e_i`` (trees per hectare represented by each tree, e.g. 1 /
    plot area in ha), giving m^2/ha.

    References
    ----------
    Avery, T. E. and Burkhart, H. E. (2002). *Forest Measurements*, 5th edn.
    McGraw-Hill, chapter 11.

    Examples
    --------
    >>> r = basal_area([20.0, 30.0], expansion=25.0)
    >>> round(r.total, 12)
    2.552544031042
    """
    d = _vec(dbh)
    e = _rep(expansion, len(d))
    g = [math.pi * v * v / 40000.0 for v in d]
    return RichResult(payload={"tree": g, "total": ssum(a * b for a, b in zip(g, e))})


def stand_density_index(dbh, expansion=1.0, method="reineke", exponent=1.605, reference=25.4):
    r"""Stand density index (trees per hectare at the reference diameter).

    ``"reineke"``: ``SDI = N (D_q / 25.4)^1.605`` with ``N = sum e_i`` trees/ha
    and the quadratic mean diameter ``D_q = sqrt(sum e_i d_i^2 / N)`` (cm);
    ``"summation"``: ``SDI = sum e_i (d_i / 25.4)^1.605`` (Zeide 1983, Shaw 2000),
    which equals Reineke's for even-aged stands of equal trees.

    References
    ----------
    Reineke, L. H. (1933). Perfecting a stand-density index for even-aged
    forests. *Journal of Agricultural Research* 46, 627-638.

    Shaw, J. D. (2000). Application of stand density index to irregularly
    structured stands. *Western Journal of Applied Forestry* 15, 40-42.

    Examples
    --------
    >>> r = stand_density_index([20.0, 30.0, 25.4], expansion=100.0)
    >>> round(r.sdi, 8), round(r.qmd, 12)
    (301.20347992, 25.46343784069)
    """
    d = _vec(dbh)
    e = _rep(expansion, len(d))
    n = ssum(e)
    qmd = math.sqrt(ssum(a * v * v for a, v in zip(e, d)) / n)
    if method == "reineke":
        sdi = n * (qmd / reference) ** exponent
    elif method == "summation":
        sdi = ssum(a * (v / reference) ** exponent for a, v in zip(e, d))
    else:
        raise ValueError("method must be 'reineke' or 'summation'")
    return RichResult(payload={"sdi": sdi, "qmd": qmd, "trees_per_ha": n})


def tree_biomass(dbh, height=None, wood_density=0.6, stress=None, model="chave2014"):
    r"""Above-ground biomass (kg) of tropical trees from the Chave et al. (2014) allometries.

    With height (m): ``AGB = 0.0673 (rho D^2 H)^0.976`` (eq 4, ``rho`` wood
    density g/cm^3, ``D`` cm). Without height, with the environmental stress
    index ``E``: ``AGB = exp(-1.803 - 0.976 E + 0.976 ln rho + 2.673 ln D -
    0.0299 (ln D)^2)`` (eq 7).

    References
    ----------
    Chave, J. et al. (2014). Improved allometric models to estimate the
    aboveground biomass of tropical trees. *Global Change Biology* 20,
    3177-3190.

    Examples
    --------
    >>> [round(v, 10) for v in tree_biomass([30.0], [22.0], 0.6)]
    [638.3162641507]
    """
    d = _vec(dbh)
    rho = _rep(wood_density, len(d))
    if model != "chave2014":
        raise ValueError("model must be 'chave2014'")
    if height is not None:
        h = _rep(height, len(d))
        return [0.0673 * (r * v * v * hh) ** 0.976 for v, hh, r in zip(d, h, rho)]
    if stress is None:
        raise ValueError("give heights or the environmental stress index E")
    E = float(stress)
    return [
        math.exp(-1.803 - 0.976 * E + 0.976 * math.log(r) + 2.673 * math.log(v) - 0.0299 * math.log(v) ** 2)
        for v, r in zip(d, rho)
    ]


def carbon_stock(biomass, carbon_fraction=0.47, root_shoot=0.0):
    r"""Carbon and CO2-equivalent from above-ground biomass.

    ``C = CF AGB (1 + R)`` with the IPCC default carbon fraction ``CF = 0.47``
    and root-to-shoot ratio ``R``; ``CO2e = C 44 / 12``.

    References
    ----------
    IPCC (2006). *Guidelines for National Greenhouse Gas Inventories*, Vol. 4,
    chapter 4.

    Examples
    --------
    >>> r = carbon_stock([1000.0, 500.0], root_shoot=0.24)
    >>> round(r.carbon, 10), round(r.co2e, 10)
    (874.2, 3205.4)
    """
    b = _vec(biomass)
    c = ssum(b) * carbon_fraction * (1.0 + root_shoot)
    return RichResult(payload={"carbon": c, "co2e": c * 44.0 / 12.0})


def rasterize_points(x, y, z, res, fun="max", extent=None):
    r"""Point-to-raster gridding of LiDAR returns (DSM with ``max``, DTM from ground returns with ``min``).

    The grid spans ``extent = (xmin, ymax, ncol, nrow)`` (default: from the
    points, snapped to multiples of ``res``); a point falls in row
    ``floor((ymax - y) / res)`` and column ``floor((x - xmin) / res)``. Each cell
    holds the ``max``, ``min`` or ``mean`` of its points, ``nan`` if empty.

    References
    ----------
    Roussel, J.-R. et al. (2020). lidR: an R package for analysis of airborne
    laser scanning data. *Remote Sensing of Environment* 251, 112061.

    Examples
    --------
    >>> r = rasterize_points([0.2, 0.7, 1.5], [0.3, 0.4, 1.2], [5.0, 7.0, 2.0], 1.0)
    >>> r.grid
    [[nan, 2.0], [7.0, nan]]
    """
    xs, ys, zs = _vec(x), _vec(y), _vec(z)
    if extent is None:
        xmin = math.floor(min(xs) / res) * res
        ymax = (math.floor(max(ys) / res) + 1) * res
        ncol = int(math.floor((max(xs) - xmin) / res)) + 1
        nrow = int(math.floor((ymax - min(ys)) / res - 1e-12)) + 1
    else:
        xmin, ymax, ncol, nrow = float(extent[0]), float(extent[1]), int(extent[2]), int(extent[3])
    acc = [[[] for _ in range(ncol)] for _ in range(nrow)]
    for a, b, v in zip(xs, ys, zs):
        r = int(math.floor((ymax - b) / res))
        c = int(math.floor((a - xmin) / res))
        if 0 <= r < nrow and 0 <= c < ncol:
            acc[r][c].append(v)
    if fun not in ("max", "min", "mean"):
        raise ValueError("fun must be 'max', 'min' or 'mean'")
    out = [
        [(max(v) if fun == "max" else min(v) if fun == "min" else ssum(v) / len(v)) if v else math.nan for v in row]
        for row in acc
    ]
    return RichResult(payload={"grid": out, "extent": [xmin, ymax, ncol, nrow], "res": res})


def canopy_height_model(dsm, dtm, floor=0.0):
    r"""Canopy height model (normalised surface model) ``CHM = max(DSM - DTM, floor)``.

    Examples
    --------
    >>> canopy_height_model([[12.0, 3.0]], [[2.0, 3.5]])
    [[10.0, 0.0]]
    """
    S, T = _grid(dsm), _grid(dtm)
    return [
        [max(a - b, floor) if not (math.isnan(a) or math.isnan(b)) else math.nan for a, b in zip(r, q)]
        for r, q in zip(S, T)
    ]


def _components(mask, connectivity):
    nr, nc = len(mask), len(mask[0])
    lab = [[0] * nc for _ in range(nr)]
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        nb += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    k = 0
    for i in range(nr):
        for j in range(nc):
            if mask[i][j] and lab[i][j] == 0:
                k += 1
                lab[i][j] = k
                stack = [(i, j)]
                while stack:
                    a, b = stack.pop()
                    for da, db in nb:
                        u, v = a + da, b + db
                        if 0 <= u < nr and 0 <= v < nc and mask[u][v] and lab[u][v] == 0:
                            lab[u][v] = k
                            stack.append((u, v))
    return lab, k


def canopy_gaps(chm, height_threshold, min_area=0.0, max_area=math.inf, res=1.0, connectivity=8):
    r"""Canopy gaps: connected cells of the CHM below a height threshold (Brokaw 1982).

    Cells with ``CHM < height_threshold`` are grouped into ``connectivity``
    (4 or 8) connected components; gaps with area (cells x ``res^2``) in
    ``[min_area, max_area]`` are kept and numbered 1, 2, ... in scan order.

    References
    ----------
    Brokaw, N. V. L. (1982). The definition of treefall gap and its effect on
    measures of forest dynamics. *Biotropica* 14, 158-160.

    Silva, C. A. et al. (2019). Comparison of small- and large-footprint lidar
    characterization of tropical forest aboveground structure and dynamics.
    *Remote Sensing* 11, 1063.

    Examples
    --------
    >>> r = canopy_gaps([[1, 9, 9], [1, 9, 2], [9, 9, 2]], 5.0, min_area=2.0)
    >>> r.labels, r.areas
    ([[1, 0, 0], [1, 0, 2], [0, 0, 2]], [2.0, 2.0])
    """
    G = _grid(chm)
    mask = [[(not math.isnan(v)) and v < height_threshold for v in r] for r in G]
    lab, k = _components(mask, connectivity)
    area = [0.0] * (k + 1)
    for row in lab:
        for v in row:
            if v:
                area[v] += res * res
    keep, new = {}, 0
    for i in range(1, k + 1):
        if min_area <= area[i] <= max_area:
            new += 1
            keep[i] = new
    labels = [[keep.get(v, 0) for v in row] for row in lab]
    areas = [area[i] for i in range(1, k + 1) if i in keep]
    total = sum(1 for r in G for v in r if not math.isnan(v)) * res * res
    return RichResult(payload={"labels": labels, "areas": areas, "gap_fraction": ssum(areas) / total})


def tree_tops(chm, res=1.0, hmin=2.0, window=None):
    r"""Tree-top detection by a local-maximum filter with a circular window.

    A cell of height ``h >= hmin`` is a tree top if no cell within the window
    radius is higher. The window diameter (m) is fixed (``window``) or varies
    with height as ``2.51503 + 0.00901 h^2`` (Popescu and Wynne 2004,
    deciduous and pine stands).

    References
    ----------
    Popescu, S. C. and Wynne, R. H. (2004). Seeing the trees in the forest:
    using lidar and multispectral data fusion with local filtering and variable
    window size for estimating tree height. *Photogrammetric Engineering and
    Remote Sensing* 70, 589-604.

    Examples
    --------
    >>> tree_tops([[1, 2, 1, 0], [2, 9, 3, 0], [1, 3, 2, 8], [0, 0, 7, 6]], hmin=2.0, window=3.0).cells
    [[1, 1], [2, 3]]
    """
    G = _grid(chm)
    nr, nc = len(G), len(G[0])
    tops, hts = [], []
    for i in range(nr):
        for j in range(nc):
            h = G[i][j]
            if math.isnan(h) or h < hmin:
                continue
            ws = float(window) if window is not None else 2.51503 + 0.00901 * h * h
            rad = ws / 2.0 / res
            k = int(math.floor(rad))
            ok = True
            for a in range(max(0, i - k), min(nr, i + k + 1)):
                for b in range(max(0, j - k), min(nc, j + k + 1)):
                    if (a - i) ** 2 + (b - j) ** 2 <= rad * rad and not math.isnan(G[a][b]) and G[a][b] > h:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                tops.append([i, j])
                hts.append(h)
    return RichResult(payload={"cells": tops, "heights": hts})


def _focal_mean(G):
    nr, nc = len(G), len(G[0])
    out = [[math.nan] * nc for _ in range(nr)]
    for i in range(nr):
        for j in range(nc):
            v = [G[a][b] for a in range(max(0, i - 1), min(nr, i + 2)) for b in range(max(0, j - 1), min(nc, j + 2))]
            v = [x for x in v if not math.isnan(x)]
            if v:
                out[i][j] = ssum(v) / len(v)
    return out


def crown_segmentation(
    chm, tops=None, search_window=3, th_seed=0.45, th_cr=0.55, dist=10.0, th=0.0, smooth=True, method="dalponte"
):
    r"""Individual tree crowns by seeded region growing (Dalponte and Coomes 2016).

    Follows the reference implementation ``itcSegment::itcIMG``: the CHM is
    optionally smoothed with a 3 x 3 mean, cells below ``th`` (or missing) are
    set to 0 and, unless ``tops`` are given (zero-based row, column), seeds are
    local maxima of a ``search_window`` square window scanned from the
    south-west corner, skipping windows that already hold a seed. Crowns then
    grow in waves: every cell added in the previous wave examines its west,
    south, north and east neighbours, and a non-zero, unlabelled neighbour of
    height ``h`` joins when ``h > th_seed h_seed``, ``h > th_cr x`` (current
    crown mean), ``h <= 1.05 h_seed`` and its Euclidean distance to the seed
    (in cells) is below ``dist``; border cells do not grow. The conditions
    use the crown mean at the time the cell is examined.
    ``method="itcsegment"`` reproduces a quirk of that package: a cell with
    exactly one qualifying neighbour does not grow (its one-row selection
    collapses to a vector and fails the size check).

    References
    ----------
    Dalponte, M. and Coomes, D. A. (2016). Tree-centric mapping of forest
    carbon density from airborne laser scanning and hyperspectral data.
    *Methods in Ecology and Evolution* 7, 1236-1245.

    Examples
    --------
    >>> chm = [[3, 5, 4, 0, 1], [6, 9, 5, 1, 2], [4, 6, 3, 7, 3], [0, 1, 6, 8, 6], [1, 2, 3, 5, 2]]
    >>> crown_segmentation(chm, [[1, 1], [3, 3]], smooth=False).labels
    [[0, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 0, 2, 0], [0, 0, 2, 2, 2], [0, 0, 0, 2, 0]]
    """
    if method not in ("dalponte", "itcsegment"):
        raise ValueError("method must be 'dalponte' or 'itcsegment'")
    Z = _grid(chm)
    if smooth:
        Z = _focal_mean(Z)
    nr, nc = len(Z), len(Z[0])
    # internal (x, y) layout of itcIMG: x = column, y = rows counted from the south edge
    G = [[0.0] * nr for _ in range(nc)]
    for r in range(nr):
        for c in range(nc):
            v = Z[r][c]
            G[c][nr - 1 - r] = 0.0 if math.isnan(v) or v < th else v
    nx, ny = nc, nr
    Idx = [[0] * ny for _ in range(nx)]
    seeds = []
    if tops is None:
        h = search_window // 2
        lo = -(-search_window // 2)
        mark = [[0] * ny for _ in range(nx)]
        for y in range(ny):
            for x in range(nx):
                if G[x][y] == 0 or not (lo <= x + 1 <= nx - lo and lo <= y + 1 <= ny - lo):
                    continue
                win = [G[a][b] for a in range(x - h, x + h + 1) for b in range(y - h, y + h + 1)]
                if (
                    G[x][y] == max(win)
                    and max(win) != 0
                    and all(mark[a][b] == 0 for a in range(x - h, x + h + 1) for b in range(y - h, y + h + 1))
                ):
                    mark[x][y] = 1
                    seeds.append((x, y))
                    Idx[x][y] = len(seeds)
    else:
        for t in tops:
            x, y = int(t[1]), nr - 1 - int(t[0])
            seeds.append((x, y))
            Idx[x][y] = len(seeds)
    crowns = [row[:] for row in Idx]
    sums = [G[x][y] for x, y in seeds]
    cnt = [1] * len(seeds)
    check = [[0] * ny for _ in range(nx)]
    old = [row[:] for row in crowns]
    grown = True
    while grown:
        grown = False
        todo = [(x, y) for y in range(ny) for x in range(nx) if crowns[x][y] != 0 and check[x][y] == 0]
        for x, y in todo:
            if x == 0 or x == nx - 1 or y == 0 or y == ny - 1:
                continue
            k = crowns[x][y]
            sx, sy = seeds[k - 1]
            hs = G[sx][sy]
            mh = sums[k - 1] / cnt[k - 1]
            ok = [
                (u, v)
                for u, v in ((x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y))
                if G[u][v] != 0
                and G[u][v] > hs * th_seed
                and G[u][v] > mh * th_cr
                and G[u][v] <= hs + hs * 0.05
                and math.sqrt((sx - u) ** 2 + (sy - v) ** 2) < dist
            ]
            if method == "itcsegment" and len(ok) < 2:
                continue
            for u, v in ok:
                if crowns[u][v] == 0:
                    crowns[u][v] = k
                    sums[k - 1] += G[u][v]
                    cnt[k - 1] += 1
                    grown = True
        check = old
        old = [row[:] for row in crowns]
    labels = [[crowns[c][nr - 1 - r] for c in range(nc)] for r in range(nr)]
    top_cells = [[nr - 1 - y, x] for x, y in seeds]
    return RichResult(
        payload={
            "labels": labels,
            "tops": top_cells,
            "crown_cells": cnt,
            "mean_height": [a / b for a, b in zip(sums, cnt)],
        }
    )


def plot_estimate(values, plot_area, N=None):
    r"""Per-hectare mean and standard error from fixed-area plots (quadrats).

    Each plot total ``y_i`` is expanded to ``y_i / a`` per hectare (``a`` plot
    area in ha); the mean and its standard error ``s / sqrt(n)`` use the finite
    population correction ``(1 - n/N)`` when the number of possible plots
    ``N`` is given.

    References
    ----------
    Cochran, W. G. (1977). *Sampling Techniques*, 3rd edn. Wiley, chapter 2.

    Examples
    --------
    >>> r = plot_estimate([3.0, 5.0, 4.0, 6.0], 0.04)
    >>> r.mean, round(r.se, 10)
    (112.5, 16.1374306092)
    """
    y = [v / plot_area for v in _vec(values)]
    n = len(y)
    m = ssum(y) / n
    s2 = ssum((v - m) ** 2 for v in y) / (n - 1)
    f = 0.0 if N is None else n / float(N)
    return RichResult(payload={"mean": m, "se": math.sqrt(s2 / n * (1 - f)), "n": n})


def stratified_estimate(values, strata, stratum_sizes):
    r"""Stratified random sampling estimator of the population mean (Cochran 1977, eqs 5.1-5.3).

    ``ybar_st = sum W_h ybar_h`` with ``W_h = N_h / N`` and ``var = sum W_h^2
    s_h^2 / n_h (1 - n_h / N_h)``; ``stratum_sizes`` maps stratum labels to
    ``N_h``.

    Examples
    --------
    >>> r = stratified_estimate([1.0, 3.0, 10.0, 14.0, 12.0], ["a", "a", "b", "b", "b"], {"a": 40, "b": 60})
    >>> round(r.mean, 12), round(r.se, 10)
    (8.0, 0.7797435476)
    """
    y = _vec(values)
    labs = list(strata)
    Ntot = float(sum(stratum_sizes.values()))
    m = 0.0
    v = 0.0
    out = {}
    for h, Nh in stratum_sizes.items():
        yh = [a for a, s in zip(y, labs) if s == h]
        nh = len(yh)
        if nh < 2:
            raise ValueError("every stratum needs at least two sampled units")
        mh = ssum(yh) / nh
        s2 = ssum((a - mh) ** 2 for a in yh) / (nh - 1)
        W = Nh / Ntot
        m += W * mh
        v += W * W * s2 / nh * (1 - nh / float(Nh))
        out[h] = mh
    return RichResult(payload={"mean": m, "se": math.sqrt(v), "stratum_means": out})


def adaptive_cluster_estimate(networks, N, ids=None):
    r"""Adaptive cluster sampling estimators of the mean per unit (Thompson 1990).

    ``networks`` lists, for each of the ``n`` initially sampled units (simple
    random sampling without replacement from ``N``), the ``y`` values of the
    network it belongs to (a unit not satisfying the condition is its own
    network). Modified Hansen-Hurwitz: ``mu_HH = mean(w_i)`` with ``w_i`` the
    network means and ``var = (N - n) / (N n (n - 1)) sum (w_i - mu_HH)^2``.
    Modified Horvitz-Thompson over the distinct networks ``k``:
    ``mu_HT = (1/N) sum y*_k / alpha_k`` with ``alpha_k = 1 - C(N - m_k, n) /
    C(N, n)``. ``ids`` names the network of each sampled unit so that a network
    hit twice is counted once in ``mu_HT`` (default: all distinct).

    References
    ----------
    Thompson, S. K. (1990). Adaptive cluster sampling. *JASA* 85, 1050-1059.

    Examples
    --------
    >>> r = adaptive_cluster_estimate([[0.0], [2.0, 6.0, 4.0], [1.0]], 20)
    >>> round(r.mean_hh, 12), round(r.mean_ht, 12)
    (1.666666666667, 1.820289855072)
    """
    nets = [_vec(v) for v in networks]
    n = len(nets)
    w = [ssum(v) / len(v) for v in nets]
    mhh = ssum(w) / n
    var = (N - n) / (N * n * (n - 1.0)) * ssum((a - mhh) ** 2 for a in w)
    seen = []
    tot = 0.0
    logc = math.lgamma(N + 1) - math.lgamma(n + 1) - math.lgamma(N - n + 1)
    keys = list(range(n)) if ids is None else list(ids)
    for v, key in zip(nets, keys):
        if key in seen:
            continue
        seen.append(key)
        mk = len(v)
        if N - mk >= n:
            lc = math.lgamma(N - mk + 1) - math.lgamma(n + 1) - math.lgamma(N - mk - n + 1)
            alpha = 1.0 - math.exp(lc - logc)
        else:
            alpha = 1.0
        tot += ssum(v) / alpha
    return RichResult(payload={"mean_hh": mhh, "se_hh": math.sqrt(var), "mean_ht": tot / N})


def line_intersect_volume(diameters, transect_length):
    r"""Downed woody debris volume by line-intersect sampling (Van Wagner 1968).

    ``V = pi^2 sum d^2 / (8 L)`` m^3/ha for intersect diameters ``d`` (cm) along
    transects of total length ``L`` (m).

    References
    ----------
    Van Wagner, C. E. (1968). The line intersect method in forest fuel
    sampling. *Forest Science* 14, 20-26.

    Examples
    --------
    >>> round(line_intersect_volume([10.0, 20.0, 15.0], 100.0), 10)
    8.9443289885
    """
    d = _vec(diameters)
    return math.pi**2 * ssum(v * v for v in d) / (8.0 * float(transect_length))


def cheatsheet() -> str:
    return (
        "basal_area / stand_density_index / tree_biomass / carbon_stock / rasterize_points / "
        "canopy_height_model / canopy_gaps / tree_tops / crown_segmentation / plot_estimate / "
        "stratified_estimate / adaptive_cluster_estimate / line_intersect_volume -> forestry."
    )
