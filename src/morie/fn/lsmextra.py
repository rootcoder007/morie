# morie.fn -- function file (rootcoder007/morie)
"""Further landscape metrics of categorical rasters (FRAGSTATS, landscapemetrics conventions): fractal,
gyration, circumscribing-circle, contiguity and core-area patch metrics, their class means and
area-weighted means, the perimeter-area fractal dimension, proximity, similarity and connectance
indices, and the information-theory metrics of Nowosad and Stepinski (2019)."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from .lsmets import _setup

__all__ = ["patch_structure", "class_structure", "proximity_metrics", "landscape_information"]


def _hull(P):
    P = sorted(set(P))
    if len(P) <= 2:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def _circ2(a, b):
    c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return c, math.hypot(a[0] - c[0], a[1] - c[1])


def _circ3(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if d == 0:
        return None
    ux = (
        (a[0] ** 2 + a[1] ** 2) * (b[1] - c[1])
        + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
        + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])
    ) / d
    uy = (
        (a[0] ** 2 + a[1] ** 2) * (c[0] - b[0])
        + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
        + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])
    ) / d
    return (ux, uy), math.hypot(a[0] - ux, a[1] - uy)


def _mec_radius(P):
    """Radius of the smallest enclosing circle (incremental Welzl on the convex hull)."""
    H = _hull(P)

    def inside(c, q):
        return math.hypot(q[0] - c[0][0], q[1] - c[0][1]) <= c[1] * (1 + 1e-12) + 1e-12

    c = (H[0], 0.0)
    for i in range(1, len(H)):
        if inside(c, H[i]):
            continue
        c = (H[i], 0.0)
        for j in range(i):
            if inside(c, H[j]):
                continue
            c = _circ2(H[i], H[j])
            for k in range(j):
                if not inside(c, H[k]):
                    c = _circ3(H[i], H[j], H[k]) or c
    return c[1]


def _core_cells(G, lab, cells, edge_depth, consider_boundary):
    """Number of core cells of each patch: cells farther than ``edge_depth`` rook steps from another patch."""
    nr, nc = len(G), len(G[0])
    core = {}
    for p, cl in cells.items():
        alive = set(cl)
        for _ in range(edge_depth):
            edge = set()
            for i, j in alive:
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    x, y = i + di, j + dj
                    if not (0 <= x < nr and 0 <= y < nc):
                        if not consider_boundary:
                            edge.add((i, j))
                    elif (x, y) not in alive:
                        edge.add((i, j))
            alive -= edge
        core[p] = len(alive)
    return core


def patch_structure(
    landscape,
    *,
    res: float = 1.0,
    directions: int = 8,
    edge_depth: int = 1,
    consider_boundary: bool = False,
    circle_method: str = "exact",
) -> RichResult:
    r"""Patch-level shape and core metrics: ``frac``, ``gyrate``, ``circle``, ``contig``, ``core`` and ``cai``.

    ``frac = 2 ln(0.25 p) / ln(a)`` (``p`` m, ``a`` m^2; 1 for single cells);
    ``gyrate`` is the mean distance (m) of the patch's cell centres from their
    centroid; ``circle = 1 - a / a_c`` with ``a_c`` the smallest circle
    enclosing the patch's cell corners; ``contig = ((sum_c t_c) / n - 1) /
    12``, where the 3x3 template gives weight 1 to the cell itself, 2 to rook
    and 1 to diagonal neighbours in the same patch; ``core`` (ha) counts
    cells more than ``edge_depth`` rook steps from any cell outside the patch
    (the landscape boundary counts unless ``consider_boundary``); ``cai =
    100 core / area``. ``circle_method="landscapemetrics"`` encloses only the
    corners of the cells in the patch's extreme rows and columns, as
    landscapemetrics' ``rcpp_get_circle`` does; that circle can miss corners of
    other cells, so the default ``"exact"`` encloses every corner. Patch order as :func:`morie.fn.lsmets.label_patches`
    (McGarigal and Marks 1995; LaGro 1991; Hesselbarth et al. 2019).

    References
    ----------
    McGarigal, K. and Marks, B. J. (1995). FRAGSTATS: spatial pattern
    analysis program for quantifying landscape structure. USDA Forest
    Service General Technical Report PNW-351.
    LaGro, J. (1991). Assessing patch shape in landscape mosaics.
    *Photogrammetric Engineering and Remote Sensing*, 57(3), 285-293.
    Hesselbarth, M. H. K., Sciaini, M., With, K. A., Wiegand, K. and
    Nowosad, J. (2019). landscapemetrics: an open-source R tool to
    calculate landscape metrics. *Ecography*, 42(10), 1648-1657.

    Examples
    --------
    >>> r = patch_structure([[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 2, 2], [2, 1, 1, 1, 1]])
    >>> [round(v, 6) for v in r.contig], [round(v, 6) for v in r.circle]
    ([0.604167, 0.0, 0.277778], [0.503126, 0.36338, 0.522535])
    >>> [round(v * 1e4, 6) for v in r.core]
    [3.0, 0.0, 0.0]
    """
    G, lab, pcls, cells, per, _ = _setup(landscape, directions)
    ids = sorted(cells)
    core = _core_cells(G, lab, cells, int(edge_depth), consider_boundary)
    frac, gyr, circ, contig, corev, cai = [], [], [], [], [], []
    for p in ids:
        cl = cells[p]
        n = len(cl)
        a = n * res * res
        pm = per[p] * res
        fr = 2 * math.log(0.25 * pm) / math.log(a) if a != 1 else math.nan
        frac.append(1.0 if fr != fr else fr)
        cx = ssum(j for _, j in cl) / n
        cy = ssum(i for i, _ in cl) / n
        gyr.append(ssum(math.hypot(j - cx, i - cy) for i, j in cl) / n * res)
        if circle_method == "exact":
            corners = {(j + dx, i + dy) for i, j in cl for dx in (0, 1) for dy in (0, 1)}
        elif circle_method == "landscapemetrics":
            r0, r1 = min(i for i, _ in cl), max(i for i, _ in cl)
            c0, c1 = min(j for _, j in cl), max(j for _, j in cl)
            corners = set()
            for i, j in cl:
                if j == c1:
                    corners |= {(j + 1, i), (j + 1, i + 1)}
                if j == c0:
                    corners |= {(j, i), (j, i + 1)}
                if i == r1:
                    corners |= {(j, i + 1), (j + 1, i + 1)}
                if i == r0:
                    corners |= {(j, i), (j + 1, i)}
        else:
            raise ValueError("circle_method must be exact or landscapemetrics")
        r = _mec_radius(list(corners)) * res
        circ.append(1 - a / (math.pi * r * r))
        s = set(cl)
        t = 0
        for i, j in cl:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if (di or dj) and (i + di, j + dj) in s:
                        t += 1 if di and dj else 2
        contig.append(((t + n) / n - 1) / 12)
        corev.append(core[p] * res * res / 10000)
        cai.append(100 * core[p] / n)
    return RichResult(
        payload={
            "id": ids,
            "patch_class": [pcls[p - 1] for p in ids],
            "frac": frac,
            "gyrate": gyr,
            "circle": circ,
            "contig": contig,
            "core": corev,
            "cai": cai,
        }
    )


def _ols_slope(x, y):
    mx, my = ssum(x) / len(x), ssum(y) / len(y)
    return ssum((a - mx) * (b - my) for a, b in zip(x, y)) / ssum((a - mx) ** 2 for a in x)


def class_structure(
    landscape,
    *,
    res: float = 1.0,
    directions: int = 8,
    edge_depth: int = 1,
    consider_boundary: bool = False,
    circle_method: str = "exact",
) -> RichResult:
    r"""Class-level means (``_mn``) and area-weighted means (``_am``) of the patch metrics, core totals and PAFRAC.

    ``frac_mn``, ``frac_am``, ``shape_am``, ``para_am``, ``gyrate_mn``,
    ``gyrate_am``, ``circle_mn``, ``contig_mn``, ``core_mn``, ``cai_mn``;
    ``tca`` total core area (ha), ``cpland = 100 tca / total area``;
    ``pafrac = 2 / b`` with ``b`` the OLS slope of ``ln a`` on ``ln p`` over
    the class's patches (``nan`` with fewer than 10 patches). The area
    weights are ``a_ij / sum_j a_ij`` (McGarigal and Marks 1995).

    Examples
    --------
    >>> c = class_structure([[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 2, 2], [2, 1, 1, 1, 1]])
    >>> [round(v, 6) for v in c.contig_mn], [round(v, 6) for v in c.gyrate_am]
    ([0.604167, 0.138889], [1.615991, 0.490529])
    """
    from .lsmets import patch_metrics

    P = patch_structure(
        landscape,
        res=res,
        directions=directions,
        edge_depth=edge_depth,
        consider_boundary=consider_boundary,
        circle_method=circle_method,
    )
    B = patch_metrics(landscape, res=res, directions=directions)
    G, *_ = _setup(landscape, directions)
    total = sum(1 for row in G for v in row if v is not None) * res * res / 10000
    classes = sorted(set(P["patch_class"]))
    out = {
        k: []
        for k in (
            "class",
            "frac_mn",
            "frac_am",
            "shape_am",
            "para_am",
            "gyrate_mn",
            "gyrate_am",
            "circle_mn",
            "contig_mn",
            "core_mn",
            "cai_mn",
            "tca",
            "cpland",
            "pafrac",
        )
    }
    for c in classes:
        ix = [k for k, v in enumerate(P["patch_class"]) if v == c]
        A = [B["area"][k] for k in ix]
        W = [a / ssum(A) for a in A]

        def mn(key, src=P, ix=ix):
            return ssum(src[key][k] for k in ix) / len(ix)

        def am(key, src=P, ix=ix, W=W):
            return ssum(w * src[key][k] for w, k in zip(W, ix))

        out["class"].append(c)
        out["frac_mn"].append(mn("frac"))
        out["frac_am"].append(am("frac"))
        out["shape_am"].append(am("shape", B))
        out["para_am"].append(am("para", B))
        out["gyrate_mn"].append(mn("gyrate"))
        out["gyrate_am"].append(am("gyrate"))
        out["circle_mn"].append(mn("circle"))
        out["contig_mn"].append(mn("contig"))
        out["core_mn"].append(mn("core"))
        out["cai_mn"].append(mn("cai"))
        tca = ssum(P["core"][k] for k in ix)
        out["tca"].append(tca)
        out["cpland"].append(100 * tca / total)
        if len(ix) < 10:
            out["pafrac"].append(math.nan)
        else:
            b = _ols_slope([math.log(B["perim"][k]) for k in ix], [math.log(B["area"][k] * 10000) for k in ix])
            out["pafrac"].append(2 / b)
    return RichResult(payload=out)


def proximity_metrics(
    landscape,
    *,
    search_radius: float,
    res: float = 1.0,
    directions: int = 8,
    similarity=None,
    threshold: float | None = None,
) -> RichResult:
    r"""FRAGSTATS proximity (PROX), similarity (SIMI) and connectance (CONNECT) indices.

    ``h_ij`` is the edge-to-edge distance between patches (smallest distance
    between their cell centres, m). ``prox_i = sum_j a_j / h_ij^2`` over the
    other patches of ``i``'s class with ``h_ij <= search_radius`` (``a`` in
    m^2); ``simi_i = sum_j a_j s(c_i, c_j) / h_ij^2`` over patches of every
    class, with ``similarity`` a class-by-class matrix (rows and columns in
    ascending class order; default identity, so ``simi = prox``).
    ``connect`` per class is ``100 sum_{j<k} [h_jk <= threshold] / (n (n -
    1) / 2)``, and the landscape value pools the joins and pairs of all
    classes; ``threshold`` defaults to ``search_radius`` (Gustafson and Parker
    1994; McGarigal and Marks 1995).

    References
    ----------
    Gustafson, E. J. and Parker, G. R. (1994). Using an index of habitat
    patch proximity for landscape design. *Landscape and Urban Planning*,
    29(2-3), 117-130.

    Examples
    --------
    >>> r = proximity_metrics([[1, 0, 1], [0, 0, 0], [1, 0, 0]], search_radius=2, directions=4)
    >>> r.prox, r.connect
    ([0.0, 0.5, 0.25, 0.25], [nan, 66.66666666666667])
    """
    G, lab, pcls, cells, per, _ = _setup(landscape, directions)
    ids = sorted(cells)
    classes = sorted(set(pcls))
    cix = {c: k for k, c in enumerate(classes)}
    if similarity is None:
        S = [[float(i == j) for j in range(len(classes))] for i in range(len(classes))]
    else:
        S = [[float(v) for v in row] for row in similarity]
    thr = search_radius if threshold is None else float(threshold)
    n = len(ids)
    H = [[0.0] * n for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            d = math.sqrt(min((i - x) ** 2 + (j - y) ** 2 for i, j in cells[ids[a]] for x, y in cells[ids[b]])) * res
            H[a][b] = H[b][a] = d
    area = [len(cells[p]) * res * res for p in ids]
    cls = [pcls[p - 1] for p in ids]
    prox, simi = [], []
    for a in range(n):
        near = [b for b in range(n) if b != a and H[a][b] <= search_radius]
        prox.append(ssum(area[b] / H[a][b] ** 2 for b in near if cls[b] == cls[a]))
        simi.append(ssum(area[b] * S[cix[cls[a]]][cix[cls[b]]] / H[a][b] ** 2 for b in near))
    conn, joins_all, pairs_all = [], 0, 0
    for c in classes:
        ix = [k for k in range(n) if cls[k] == c]
        m = len(ix)
        joins = sum(1 for u in range(m) for v in range(u + 1, m) if H[ix[u]][ix[v]] <= thr)
        pairs = m * (m - 1) // 2
        conn.append(100 * joins / pairs if pairs else math.nan)
        joins_all += joins
        pairs_all += pairs
    return RichResult(
        payload={
            "id": ids,
            "patch_class": cls,
            "prox": prox,
            "simi": simi,
            "class": classes,
            "connect": conn,
            "connect_landscape": 100 * joins_all / pairs_all if pairs_all else math.nan,
        }
    )


def _entropy(counts, base):
    tot = ssum(counts)
    lg = {"log2": math.log2, "log": math.log, "log10": math.log10}[base]
    return -ssum(c / tot * lg(c / tot) for c in counts if c > 0)


def landscape_information(landscape, *, base: str = "log2") -> RichResult:
    r"""Information-theory landscape metrics from the rook co-occurrence matrix (Nowosad and Stepinski 2019).

    With ``C`` the matrix of ordered rook-adjacent class pairs (each
    adjacency counted in both directions, ``NA`` cells ignored): ``ent =
    H(X)`` of its column sums (marginal entropy), ``joinent = H(X, Y)``,
    ``condent = H(Y | X) = joinent - ent``, ``mutinf = ent - condent`` and
    ``relmutinf = mutinf / ent`` (1 when ``mutinf = 0``, as landscapemetrics).

    References
    ----------
    Nowosad, J. and Stepinski, T. F. (2019). Information theory as a
    consistent framework for quantification and classification of landscape
    patterns. *Landscape Ecology*, 34(9), 2091-2101.

    Examples
    --------
    >>> r = landscape_information([[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 2, 2], [2, 1, 1, 1, 1]])
    >>> round(r.ent, 6), round(r.condent, 6), round(r.mutinf, 6)
    (0.708836, 0.689275, 0.019561)
    """
    G, *_ = _setup(landscape, 4)
    nr, nc = len(G), len(G[0])
    C = {}
    for i in range(nr):
        for j in range(nc):
            if G[i][j] is None:
                continue
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                x, y = i + di, j + dj
                if 0 <= x < nr and 0 <= y < nc and G[x][y] is not None:
                    C[(G[i][j], G[x][y])] = C.get((G[i][j], G[x][y]), 0) + 1
    classes = sorted({k[1] for k in C})
    colsum = [ssum(v for k, v in C.items() if k[1] == c) for c in classes]
    ent = _entropy(colsum, base)
    joinent = _entropy(list(C.values()), base)
    condent = joinent - ent
    mutinf = ent - condent
    return RichResult(
        payload={
            "ent": ent,
            "joinent": joinent,
            "condent": condent,
            "mutinf": mutinf,
            "relmutinf": 1.0 if mutinf == 0 else mutinf / ent,
        }
    )


def cheatsheet() -> str:
    return (
        "patch_structure / class_structure / proximity_metrics / landscape_information -> FRAGSTATS "
        "frac, gyrate, circle, contig, core, pafrac, prox, simi, connect and information-theory metrics."
    )
