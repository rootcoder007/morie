# morie.fn -- function file (rootcoder007/morie)
"""Multiple-point geostatistical simulation from a training image: direct sampling (Mariethoz,
Renard and Straubhaar 2010) and single-grid SNESIM (Strebelle 2002). Grids are lists of rows;
``nan`` marks unknown nodes; the random path, the scan start and the category draws come from the
Philox stream (``seed``)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["direct_sampling", "snesim"]


def _grid(g):
    return [[float(v) for v in r] for r in np.asarray(g, dtype=float).tolist()]


def _setup(nx, ny, conditioning):
    sg = [[math.nan] * nx for _ in range(ny)]
    for i, j, v in conditioning or []:
        sg[int(i)][int(j)] = float(v)
    return sg


def _path(sg, u):
    free = [(i, j) for i in range(len(sg)) for j in range(len(sg[0])) if math.isnan(sg[i][j])]
    order = sorted(range(len(free)), key=lambda k: (u[k], k))
    return [free[k] for k in order]


def direct_sampling(
    ti, nx, ny, n_neighbors=12, threshold=0.05, max_fraction=0.5, conditioning=None, categorical=True, seed=0
):
    r"""Direct sampling multiple-point simulation.

    Nodes are visited along a random path. The data event of a node is formed
    by its ``n_neighbors`` closest informed nodes (lags ``h_k``, ties by scan
    order). The training image is scanned in raster order from a random start
    until a location ``y`` whose pattern ``TI(y + h_k)`` is within ``threshold``
    of the data event (mismatch fraction for ``categorical``, otherwise the
    root-mean-square difference over the training-image range), or until
    ``max_fraction`` of it has been scanned, in which case the closest pattern
    found is used; the node takes ``TI(y)``. ``conditioning`` lists ``(row,
    col, value)`` hard data.

    References
    ----------
    Mariethoz, G., Renard, P. and Straubhaar, J. (2010). The direct sampling
    method to perform multiple-point geostatistical simulations. *Water
    Resources Research* 46, W11536.

    Examples
    --------
    >>> ti = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
    >>> r = direct_sampling(ti, 4, 3, conditioning=[(0, 0, 0.0), (0, 3, 1.0)], seed=1)
    >>> r.grid[0][0], r.grid[0][3], all(v in (0.0, 1.0) for row in r.grid for v in row)
    (0.0, 1.0, True)
    """
    T = _grid(ti)
    tr, tc = len(T), len(T[0])
    sg = _setup(nx, ny, conditioning)
    ntot = nx * ny
    u = [float(v) for v in random_uniform(2 * ntot + 1, seed=seed)]
    path = _path(sg, u)
    tvals = [v for r in T for v in r]
    rng = (max(tvals) - min(tvals)) or 1.0
    informed = [(i, j) for i in range(ny) for j in range(nx) if not math.isnan(sg[i][j])]
    maxscan = max(1, int(max_fraction * tr * tc))
    for step, (i, j) in enumerate(path):
        start = int(u[ntot + step] * tr * tc)
        near = sorted(informed, key=lambda p: ((p[0] - i) ** 2 + (p[1] - j) ** 2, p[0], p[1]))[:n_neighbors]
        ev = [(p[0] - i, p[1] - j, sg[p[0]][p[1]]) for p in near]
        best, best_d = None, math.inf
        for s in range(maxscan):
            pos = (start + s) % (tr * tc)
            a, b = divmod(pos, tc)
            if any(not (0 <= a + di < tr and 0 <= b + dj < tc) for di, dj, _ in ev):
                continue
            if not ev:
                best = (a, b)
                break
            if categorical:
                d = sum(1 for di, dj, v in ev if T[a + di][b + dj] != v) / len(ev)
            else:
                d = math.sqrt(sum((T[a + di][b + dj] - v) ** 2 for di, dj, v in ev) / len(ev)) / rng
            if d < best_d:
                best, best_d = (a, b), d
            if d <= threshold:
                break
        if best is None:
            a, b = divmod(start, tc)
            best = (a, b)
        sg[i][j] = T[best[0]][best[1]]
        informed.append((i, j))
    return RichResult(payload={"grid": sg, "path": path})


def snesim(ti, nx, ny, radius=2, max_nodes=12, min_replicates=5, conditioning=None, seed=0):
    r"""Single-grid SNESIM simulation of a categorical variable.

    The template holds the offsets within a square of half-width ``radius``
    (closest ``max_nodes`` by distance, ties in scan order). At each node of a
    random path the data event is formed by the informed template nodes; the
    training-image locations matching it are counted by category, and the
    farthest event node is dropped while fewer than ``min_replicates`` matches
    remain. The category is drawn from the resulting conditional proportions
    (the training-image marginal when nothing matches). This is the search-tree
    probability of Strebelle (2002) computed by direct scanning, without
    multiple grids.

    References
    ----------
    Strebelle, S. (2002). Conditional simulation of complex geological
    structures using multiple-point statistics. *Mathematical Geology* 34,
    1-21.

    Examples
    --------
    >>> ti = [[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0]]
    >>> r = snesim(ti, 5, 4, radius=1, conditioning=[(0, 0, 1.0)], seed=2)
    >>> r.grid[0][0], sorted(r.categories)
    (1.0, [0.0, 1.0])
    """
    T = _grid(ti)
    tr, tc = len(T), len(T[0])
    cats = sorted({v for r in T for v in r})
    marg = [sum(1 for r in T for v in r if v == c) / (tr * tc) for c in cats]
    tmpl = sorted(
        ((di, dj) for di in range(-radius, radius + 1) for dj in range(-radius, radius + 1) if (di, dj) != (0, 0)),
        key=lambda h: (h[0] ** 2 + h[1] ** 2, h[0], h[1]),
    )[:max_nodes]
    sg = _setup(nx, ny, conditioning)
    ntot = nx * ny
    u = [float(v) for v in random_uniform(2 * ntot + 1, seed=seed)]
    path = _path(sg, u)
    for step, (i, j) in enumerate(path):
        ev = [
            (di, dj, sg[i + di][j + dj])
            for di, dj in tmpl
            if 0 <= i + di < ny and 0 <= j + dj < nx and not math.isnan(sg[i + di][j + dj])
        ]
        while True:
            cnt = [0] * len(cats)
            for a in range(tr):
                for b in range(tc):
                    if all(0 <= a + di < tr and 0 <= b + dj < tc and T[a + di][b + dj] == v for di, dj, v in ev):
                        cnt[cats.index(T[a][b])] += 1
            if sum(cnt) >= min_replicates or not ev:
                break
            ev = ev[:-1]
        tot = sum(cnt)
        prob = [c / tot for c in cnt] if tot > 0 else marg
        x = u[ntot + step]
        acc, k = 0.0, len(cats) - 1
        for m, pm in enumerate(prob):
            acc += pm
            if x < acc:
                k = m
                break
        sg[i][j] = cats[k]
    return RichResult(payload={"grid": sg, "categories": cats, "template": tmpl})


def cheatsheet() -> str:
    return "direct_sampling / snesim -> multiple-point simulation."
