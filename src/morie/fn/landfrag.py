# morie.fn -- function file (rootcoder007/morie)
"""Landscape shape and fragmentation indices: the dissection (shape) index of a patch or basin and
the Riitters et al. (2000) moving-window forest fragmentation model (interior, perforated, edge,
transitional, patch and undetermined forest)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["dissection_index", "forest_fragmentation"]


def dissection_index(area, perimeter):
    r"""Dissection index ``DI = P / (2 sqrt(pi A))``: perimeter relative to a circle of equal area.

    1 for a circle and increasing with boundary dissection (Patton 1975's
    diversity index; used for landform and patch dissection).

    References
    ----------
    Patton, D. R. (1975). A diversity index for quantifying habitat "edge".
    *Wildlife Society Bulletin* 3, 171-173.

    Examples
    --------
    >>> round(dissection_index(100.0, 40.0), 12)
    1.128379167096
    """
    if isinstance(area, (int, float)):
        return perimeter / (2 * math.sqrt(math.pi * area))
    a = [float(v) for v in np.asarray(area, dtype=float).ravel().tolist()]
    p = [float(v) for v in np.asarray(perimeter, dtype=float).ravel().tolist()]
    return [q / (2 * math.sqrt(math.pi * s)) for s, q in zip(a, p)]


def forest_fragmentation(grid, window=3):
    r"""Riitters et al. (2000) forest fragmentation classes from a binary forest map.

    For each forest cell, in the ``window`` x ``window`` neighbourhood (cells
    off the map ignored): ``Pf`` is the forest proportion and ``Pff`` the share
    of cardinally adjacent cell pairs with at least one forest cell that are
    both forest. Classes: ``interior`` (``Pf = 1``), ``patch`` (``Pf < 0.4``),
    ``transitional`` (``0.4 <= Pf < 0.6``), and for ``Pf >= 0.6``: ``edge``
    (``Pf - Pff < 0``), ``perforated`` (``Pf - Pff > 0``) or ``undetermined``
    (``Pf = Pff``). Non-forest cells are class ``None``; ``nan`` cells are
    missing.

    References
    ----------
    Riitters, K., Wickham, J., O'Neill, R., Jones, B. and Smith, E. (2000).
    Global-scale patterns of forest fragmentation. *Conservation Ecology*
    4(2), 3.

    Examples
    --------
    >>> r = forest_fragmentation([[1, 1, 1], [1, 1, 1], [1, 1, 0]])
    >>> r.classes[0][0], r.classes[1][1]
    ('interior', 'perforated')
    """
    G = [[float(v) for v in r] for r in np.asarray(grid, dtype=float).tolist()]
    nr, nc = len(G), len(G[0])
    h = window // 2
    cls = [[None] * nc for _ in range(nr)]
    pf_g = [[math.nan] * nc for _ in range(nr)]
    pff_g = [[math.nan] * nc for _ in range(nr)]
    for i in range(nr):
        for j in range(nc):
            if G[i][j] != 1.0:
                continue
            cells = [
                (a, b)
                for a in range(max(0, i - h), min(nr, i + h + 1))
                for b in range(max(0, j - h), min(nc, j + h + 1))
                if not math.isnan(G[a][b])
            ]
            inside = set(cells)
            pf = sum(1 for a, b in cells if G[a][b] == 1.0) / len(cells)
            any_f = both = 0
            for a, b in cells:
                for u, v in ((a + 1, b), (a, b + 1)):
                    if (u, v) in inside:
                        f1, f2 = G[a][b] == 1.0, G[u][v] == 1.0
                        if f1 or f2:
                            any_f += 1
                            both += f1 and f2
            pff = both / any_f if any_f else math.nan
            if pf == 1.0:
                c = "interior"
            elif pf < 0.4:
                c = "patch"
            elif pf < 0.6:
                c = "transitional"
            elif pf - pff < 0:
                c = "edge"
            elif pf - pff > 0:
                c = "perforated"
            else:
                c = "undetermined"
            cls[i][j] = c
            pf_g[i][j] = pf
            pff_g[i][j] = pff
    counts = {}
    for row in cls:
        for c in row:
            if c is not None:
                counts[c] = counts.get(c, 0) + 1
    return RichResult(payload={"classes": cls, "pf": pf_g, "pff": pff_g, "counts": counts})


def cheatsheet() -> str:
    return "dissection_index / forest_fragmentation -> landscape fragmentation."
