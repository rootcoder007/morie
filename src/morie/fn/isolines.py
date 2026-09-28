# morie.fn -- function file (rootcoder007/morie)
"""Contour (iso-)lines of a gridded surface by marching squares, with lengths per level."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = ["iso_lines"]

# edges of cell (i, j): 0 bottom (y_j, x_i..x_i+1), 1 right, 2 top, 3 left; corners bl, br, tr, tl
_SEG = {
    1: [(3, 0)],
    2: [(0, 1)],
    3: [(3, 1)],
    4: [(1, 2)],
    6: [(0, 2)],
    7: [(3, 2)],
    8: [(2, 3)],
    9: [(0, 2)],
    11: [(1, 2)],
    12: [(1, 3)],
    13: [(0, 1)],
    14: [(3, 0)],
}


def iso_lines(x, y, z, levels) -> RichResult:
    r"""Iso-lines of ``z`` (``len(x)`` rows by ``len(y)`` columns, ``z[i][j]`` at ``(x_i, y_j)``) by marching squares.

    A corner is "above" when ``z > level``; crossings are linearly
    interpolated along cell edges (as ``grDevices::contourLines``), each edge
    crossing is computed once so neighbouring cells share it exactly, and
    segments are chained into polylines (closed rings repeat their first
    vertex). Saddle cells are resolved by the mean of the four corners (above
    the level: the two above-corners are connected). Cells with a missing
    corner are skipped. Returns, per level, the polylines and their total
    length.

    References
    ----------
    Lorensen, W. E. and Cline, H. E. (1987). Marching cubes: a high
    resolution 3D surface construction algorithm. *Computer Graphics*,
    21(4), 163-169.

    Examples
    --------
    >>> r = iso_lines([0, 1, 2], [0, 1, 2], [[0, 0, 0], [0, 2, 0], [0, 0, 0]], [1.0])
    >>> len(r.lines), round(r.length[0], 6), r.lines[0]["x"][0] == r.lines[0]["x"][-1]
    (1, 2.828427, True)
    """
    X, Y = [float(v) for v in x], [float(v) for v in y]
    Z = [[float(v) for v in row] for row in np.asarray(z, dtype=float).tolist()]
    nx, ny = len(X), len(Y)
    if len(Z) != nx or any(len(r) != ny for r in Z):
        raise ValueError("z must be len(x) x len(y)")
    lines, lengths = [], []
    for L in (float(v) for v in levels):
        cache = {}

        def pt(a, b, L=L, cache=cache):
            key = (a, b) if a < b else (b, a)
            if key not in cache:
                (i1, j1), (i2, j2) = key
                z1, z2 = Z[i1][j1], Z[i2][j2]
                t = (L - z1) / (z2 - z1)
                cache[key] = (X[i1] + t * (X[i2] - X[i1]), Y[j1] + t * (Y[j2] - Y[j1]))
            return key

        segs = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                c = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                v = [Z[a][b] for a, b in c]
                if any(w != w for w in v):
                    continue
                code = sum(1 << k for k in range(4) if v[k] > L)
                edges = {0: (c[0], c[1]), 1: (c[1], c[2]), 2: (c[3], c[2]), 3: (c[0], c[3])}
                if code in (5, 10):
                    centre_above = ssum(v) / 4 > L
                    if centre_above:
                        pairs = [(3, 2), (0, 1)] if code == 5 else [(0, 3), (2, 1)]
                    else:
                        pairs = [(3, 0), (1, 2)] if code == 5 else [(0, 1), (2, 3)]
                else:
                    pairs = _SEG.get(code, [])
                for e1, e2 in pairs:
                    segs.append((pt(*edges[e1]), pt(*edges[e2])))
        adj = {}
        for k, (a, b) in enumerate(segs):
            adj.setdefault(a, []).append(k)
            adj.setdefault(b, []).append(k)
        used = [False] * len(segs)
        level_lines = []
        for k0 in range(len(segs)):
            if used[k0]:
                continue
            used[k0] = True
            chain = [segs[k0][0], segs[k0][1]]
            for end in (1, 0):
                while True:
                    tip = chain[-1] if end else chain[0]
                    nxt = next((k for k in adj[tip] if not used[k]), None)
                    if nxt is None:
                        break
                    used[nxt] = True
                    a, b = segs[nxt]
                    other = b if a == tip else a
                    if end:
                        chain.append(other)
                    else:
                        chain.insert(0, other)
            P = [cache[q] for q in chain]
            level_lines.append({"level": L, "x": [p[0] for p in P], "y": [p[1] for p in P]})
        lines.extend(level_lines)
        lengths.append(
            ssum(
                math.dist((ln["x"][t], ln["y"][t]), (ln["x"][t + 1], ln["y"][t + 1]))
                for ln in level_lines
                for t in range(len(ln["x"]) - 1)
            )
        )
    return RichResult(payload={"lines": lines, "length": lengths, "levels": [float(v) for v in levels]})


def cheatsheet() -> str:
    return "iso_lines(x, y, z, levels) -> marching-squares contour lines and lengths."
