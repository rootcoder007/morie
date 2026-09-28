# morie.fn -- function file (rootcoder007/morie)
"""Landscape metrics of categorical rasters (FRAGSTATS definitions, landscapemetrics conventions)."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = ["label_patches", "patch_metrics", "class_metrics", "landscape_metrics"]


def _grid(landscape):
    G = [[None if (v is None or (isinstance(v, float) and v != v)) else int(v) for v in row] for row in landscape]
    if not G or any(len(r) != len(G[0]) for r in G):
        raise ValueError("landscape must be a rectangular 2-D grid")
    return G


def label_patches(landscape, directions: int = 8) -> RichResult:
    r"""Patches: connected components of equal-class cells (``directions`` 4 or 8 neighbours).

    Patches are numbered from 1 by class (ascending) and, within a class,
    in column-major order of their first cell, as ``landscapemetrics::get_patches``.

    Examples
    --------
    >>> label_patches([[1, 1, 2], [2, 1, 2]], directions=4).labels
    [[1, 1, 3], [2, 1, 3]]
    """
    if directions not in (4, 8):
        raise ValueError("directions must be 4 or 8")
    G = _grid(landscape)
    nr, nc = len(G), len(G[0])
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if directions == 8 else [])
    lab = [[0] * nc for _ in range(nr)]
    cls = []
    nxt = 0
    for k in sorted({v for row in G for v in row if v is not None}):
        for j in range(nc):
            for i in range(nr):
                if G[i][j] != k or lab[i][j]:
                    continue
                nxt += 1
                cls.append(k)
                lab[i][j] = nxt
                stack = [(i, j)]
                while stack:
                    a, b = stack.pop()
                    for da, db in nb:
                        x, y = a + da, b + db
                        if 0 <= x < nr and 0 <= y < nc and not lab[x][y] and G[x][y] == k:
                            lab[x][y] = nxt
                            stack.append((x, y))
    return RichResult(payload={"labels": lab, "patch_class": cls})


def _min_edges(n):
    """Minimum number of cell edges bounding ``n`` cells (FRAGSTATS)."""
    m = math.isqrt(n)
    if m * m == n:
        return 4 * m
    if n <= m * (m + 1):
        return 4 * m + 2
    return 4 * m + 4


def _max_like(n):
    """Maximum number of like (rook) adjacencies among ``n`` cells, single count (FRAGSTATS AI)."""
    m = math.isqrt(n)
    r = n - m * m
    if r == 0:
        return 2 * m * (m - 1)
    if r <= m:
        return 2 * m * (m - 1) + 2 * r - 1
    return 2 * m * (m - 1) + 2 * r - 2


def _setup(landscape, directions):
    G = _grid(landscape)
    L = label_patches(G, directions)
    lab, pcls = L["labels"], L["patch_class"]
    nr, nc = len(G), len(G[0])
    cells = {}
    for i in range(nr):
        for j in range(nc):
            if lab[i][j]:
                cells.setdefault(lab[i][j], []).append((i, j))
    # rook edges of each patch: to other classes, to NA, to the outside
    per = {p: 0 for p in cells}
    adj = {}  # (class, class) rook adjacencies, double count, including like
    for i in range(nr):
        for j in range(nc):
            if G[i][j] is None:
                continue
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                x, y = i + di, j + dj
                inside = 0 <= x < nr and 0 <= y < nc
                other = G[x][y] if inside else None
                if not inside or other is None or other != G[i][j]:
                    per[lab[i][j]] += 1
                if inside and other is not None:
                    adj[(G[i][j], other)] = adj.get((G[i][j], other), 0) + 1
    return G, lab, pcls, cells, per, adj


def patch_metrics(landscape, *, res: float = 1.0, directions: int = 8) -> RichResult:
    r"""Patch-level metrics: ``area`` (ha), ``perim`` (m), ``shape``, ``para`` and ``enn`` (m).

    ``area = n res^2 / 10^4``; ``perim`` counts every cell edge of the patch
    not shared with a cell of the same patch (landscape boundary included);
    ``shape = 0.25 perim / sqrt(n res^2)`` (landscapemetrics; FRAGSTATS'
    vector form); ``para = perim / (n res^2)``; ``enn`` the smallest distance
    between cell centres of the patch and any other patch of its class
    (``nan`` if none) (McGarigal and Marks 1995; Hesselbarth et al. 2019).

    References
    ----------
    McGarigal, K. and Marks, B. J. (1995). FRAGSTATS: spatial pattern
    analysis program for quantifying landscape structure. USDA Forest
    Service General Technical Report PNW-351.
    Hesselbarth, M. H. K., Sciaini, M., With, K. A., Wiegand, K. and
    Nowosad, J. (2019). landscapemetrics: an open-source R tool to
    calculate landscape metrics. *Ecography*, 42(10), 1648-1657.

    Examples
    --------
    >>> r = patch_metrics([[1, 1, 2], [2, 1, 2]], directions=4)
    >>> r.patch_class, [round(v, 6) for v in r.shape]
    ([1, 2, 2], [1.154701, 1.0, 1.06066])
    """
    G, lab, pcls, cells, per, adj = _setup(landscape, directions)
    ids = sorted(cells)
    area = [len(cells[p]) * res * res / 10000.0 for p in ids]
    perim = [per[p] * res for p in ids]
    shape = [0.25 * per[p] / math.sqrt(len(cells[p])) for p in ids]
    para = [per[p] * res / (len(cells[p]) * res * res) for p in ids]
    enn = []
    for p in ids:
        same = [q for q in ids if q != p and pcls[q - 1] == pcls[p - 1]]
        if not same:
            enn.append(float("nan"))
            continue
        best = min((a - c) ** 2 + (b - d) ** 2 for q in same for a, b in cells[p] for c, d in cells[q])
        enn.append(math.sqrt(best) * res)
    return RichResult(
        payload={
            "id": ids,
            "patch_class": [pcls[p - 1] for p in ids],
            "area": area,
            "perim": perim,
            "shape": shape,
            "para": para,
            "enn": enn,
        }
    )


def class_metrics(landscape, *, res: float = 1.0, directions: int = 8) -> RichResult:
    r"""Class-level metrics, one value per class (ascending).

    ``ca`` (ha), ``pland`` (%), ``np``, ``pd`` (patches per 100 ha), ``lpi``
    (%), ``ed`` (edge to other classes per ha, landscape boundary excluded),
    ``lsi = e / e_min`` (edges incl. boundary over the minimum for the class
    area), ``ai = 100 g_ii / g_max``, ``clumpy``, ``cohesion``, ``iji``,
    ``area_mn``, ``shape_mn``, ``enn_mn`` (McGarigal and Marks 1995;
    FRAGSTATS 4 help; landscapemetrics).

    Examples
    --------
    >>> c = class_metrics([[1, 1, 2], [2, 1, 2]], directions=4)
    >>> c.np, [round(v, 6) for v in c.pland]
    ([1, 2], [50.0, 50.0])
    """
    G, lab, pcls, cells, per, adj = _setup(landscape, directions)
    P = patch_metrics(G, res=res, directions=directions)
    classes = sorted(set(pcls))
    nr, nc = len(G), len(G[0])
    outer = {}  # edges of each class to the outside of the grid or to NA cells
    for i in range(nr):
        for j in range(nc):
            if G[i][j] is not None:
                for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if not (0 <= x < nr and 0 <= y < nc) or G[x][y] is None:
                        outer[G[i][j]] = outer.get(G[i][j], 0) + 1
    Z = sum(1 for row in G for v in row if v is not None)
    ta = Z * res * res / 10000.0
    out = {
        k: []
        for k in (
            "class",
            "ca",
            "pland",
            "np",
            "pd",
            "lpi",
            "ed",
            "lsi",
            "ai",
            "clumpy",
            "cohesion",
            "iji",
            "area_mn",
            "shape_mn",
            "enn_mn",
        )
    }
    for k in classes:
        idx = [t for t, c in enumerate(P["patch_class"]) if c == k]
        n = sum(len(cells[P["id"][t]]) for t in idx)
        like = adj.get((k, k), 0)  # double count
        unlike = ssum(v for (a, b), v in adj.items() if a == k and b != k)
        tot_e = ssum(per[P["id"][t]] for t in idx)
        pk = n / Z
        out["class"].append(k)
        out["ca"].append(n * res * res / 10000.0)
        out["pland"].append(100.0 * pk)
        out["np"].append(len(idx))
        out["pd"].append(len(idx) / ta * 100.0)
        out["lpi"].append(100.0 * max(P["area"][t] for t in idx) / ta)
        out["ed"].append(unlike * res / (ta * 10000.0) * 10000.0)
        out["lsi"].append(tot_e / _min_edges(n))
        gm = _max_like(n)
        out["ai"].append(100.0 * (like / 2) / gm if gm > 0 else float("nan"))
        # clumpiness: G_i = g_ii / (sum_k g_ik - min_e_i), like adjacencies double counted and edges to the
        # outside or to NA counted (landscapemetrics pads the raster); FRAGSTATS' CLUMPY from G_i and P_i
        den = like + unlike + outer.get(k, 0) - _min_edges(n)
        Gi = like / den if den else float("nan")
        if Gi != Gi or pk == 1.0:
            out["clumpy"].append(float("nan"))
        else:
            out["clumpy"].append((Gi - pk) / pk if (Gi < pk and pk < 0.5) else (Gi - pk) / (1.0 - pk))
        pp = [per[P["id"][t]] for t in idx]
        aa = [len(cells[P["id"][t]]) for t in idx]
        out["cohesion"].append(
            100.0 * (1.0 - ssum(pp) / ssum(p * math.sqrt(a) for p, a in zip(pp, aa))) / (1.0 - 1.0 / math.sqrt(Z))
        )
        others = [b for b in classes if b != k]
        e_ik = [adj.get((k, b), 0) for b in others]
        E = ssum(e_ik)
        m = len(classes)
        if m < 3 or E == 0:
            out["iji"].append(float("nan"))
        else:
            out["iji"].append(-100.0 * ssum((e / E) * math.log(e / E) for e in e_ik if e > 0) / math.log(m - 1))
        out["area_mn"].append(ssum(P["area"][t] for t in idx) / len(idx))
        out["shape_mn"].append(ssum(P["shape"][t] for t in idx) / len(idx))
        en = [P["enn"][t] for t in idx if P["enn"][t] == P["enn"][t]]
        out["enn_mn"].append(ssum(en) / len(en) if en else float("nan"))
    return RichResult(payload=out)


def landscape_metrics(landscape, *, res: float = 1.0, directions: int = 8) -> RichResult:
    r"""Landscape-level metrics.

    ``ta`` (ha), ``np``, ``pd``, ``lpi``, ``ed``, ``lsi``, ``ai`` (area
    weighted over classes of like adjacencies), ``contag`` (Li and Reynolds
    1993 with double-counted rook adjacencies), ``iji``, ``pr`` (richness),
    ``shdi``, ``shei``, ``sidi``, ``siei`` and ``dominance`` ``ln(pr) -
    shdi`` (O'Neill et al. 1988), ``shape_mn``, ``area_mn``, ``enn_mn``.

    References
    ----------
    Li, H. and Reynolds, J. F. (1993). A new contagion index to quantify
    spatial patterns of landscapes. *Landscape Ecology*, 8(3), 155-162.
    O'Neill, R. V. et al. (1988). Indices of landscape pattern. *Landscape
    Ecology*, 1(3), 153-162.

    Examples
    --------
    >>> round(landscape_metrics([[1, 1, 2], [2, 1, 2]], directions=4).shdi, 6)
    0.693147
    """
    G, lab, pcls, cells, per, adj = _setup(landscape, directions)
    P = patch_metrics(G, res=res, directions=directions)
    C = class_metrics(G, res=res, directions=directions)
    classes = C["class"]
    m = len(classes)
    Z = sum(1 for row in G for v in row if v is not None)
    ta = Z * res * res / 10000.0
    p = [v / 100.0 for v in C["pland"]]
    shdi = -ssum(v * math.log(v) for v in p if v > 0)
    sidi = 1.0 - ssum(v * v for v in p)
    unlike_total = ssum(v for (a, b), v in adj.items() if a != b) / 2
    like = [adj.get((k, k), 0) / 2 for k in classes]
    gmax = [_max_like(round(v * Z)) for v in p]
    gt = ssum(adj.values())
    cont = ssum((g / gt) * math.log(g / gt) for g in adj.values() if g > 0)
    e_pairs = {}
    for (a, b), v in adj.items():
        if a < b:
            e_pairs[(a, b)] = v
    E = ssum(e_pairs.values())
    iji = float("nan")
    if m >= 3 and E > 0:
        iji = -100.0 * ssum((e / E) * math.log(e / E) for e in e_pairs.values() if e > 0) / math.log(m * (m - 1) / 2)
    en = [v for v in P["enn"] if v == v]
    out = {
        "ta": ta,
        "np": len(P["id"]),
        "pd": len(P["id"]) / ta * 100.0,
        "lpi": 100.0 * max(P["area"]) / ta,
        "ed": unlike_total * res / (ta * 10000.0) * 10000.0,
        "lsi": (unlike_total + _perimeter_outer(G)) / _min_edges(Z),
        "ai": 100.0 * ssum(pi * (li / g) for pi, li, g in zip(p, like, gmax) if g > 0),
        "contag": 100.0 * (1.0 + cont / (2.0 * math.log(m))) if m > 1 else float("nan"),
        "iji": iji,
        "pr": m,
        "shdi": shdi,
        "shei": shdi / math.log(m) if m > 1 else float("nan"),
        "sidi": sidi,
        "siei": sidi / (1.0 - 1.0 / m) if m > 1 else float("nan"),
        "dominance": math.log(m) - shdi,
        "shape_mn": ssum(P["shape"]) / len(P["shape"]),
        "area_mn": ssum(P["area"]) / len(P["area"]),
        "enn_mn": ssum(en) / len(en) if en else float("nan"),
    }
    return RichResult(payload=out)


def _perimeter_outer(G):
    """Edges between data cells and NA or the outside of the grid."""
    nr, nc = len(G), len(G[0])
    e = 0
    for i in range(nr):
        for j in range(nc):
            if G[i][j] is None:
                continue
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                x, y = i + di, j + dj
                if not (0 <= x < nr and 0 <= y < nc) or G[x][y] is None:
                    e += 1
    return e


def cheatsheet() -> str:
    return "patch_metrics / class_metrics / landscape_metrics -> FRAGSTATS landscape metrics (landscapemetrics)."
