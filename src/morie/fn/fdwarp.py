# morie.fn -- function file (rootcoder007/morie)
"""Dynamic time warping.

Sakoe, H. and Chiba, S. (1978), "Dynamic programming algorithm
optimization for spoken word recognition", *IEEE Transactions on
Acoustics, Speech, and Signal Processing* 26(1), 43-49.  The paper
defines the time-normalised distance between two patterns as the
minimum, over monotone alignment paths, of the accumulated local
distance, and computes it by the forward dynamic-programming recursion

    g(i, j) = d(i, j) + min[ g(i-1, j), g(i-1, j-1), g(i, j-1) ],

with g(1, 1) = d(1, 1).  That is the symmetric form with no slope
constraint, equation (7) of the paper in its unweighted case, and it is
the "DP over alignment cost" named in the stub docstring.

The Sakoe-Chiba adjustment window |i - j| <= window is supported and
is off by default.  The time-normalised distance divides the
accumulated cost by the length of the optimal path, which for this
symmetric step pattern is the path length itself rather than a fixed
normalisation.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["functional_warping"]

_INF = float("inf")


def functional_warping(x, y, cost="abs", window=None):
    """Dynamic time warping distance and optimal path.

    Parameters
    ----------
    x, y : array-like
        The two sequences; they need not have the same length.
    cost : str
        Local distance, "abs" for |x_i - y_j| or "sq" for its square.
    window : int, optional
        Sakoe-Chiba adjustment window; unrestricted when omitted.

    Returns
    -------
    estimate : the accumulated DTW distance
    distance : the same
    normalized : distance divided by the optimal path length
    path_length : number of cells on the optimal path
    path : the optimal path as (i, j) index pairs, from (0, 0)
    """
    a = k.vec(x)
    b = k.vec(y)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("functional_warping: both sequences must be non-empty")
    if cost not in ("abs", "sq"):
        raise ValueError("functional_warping: cost must be abs or sq")
    if window is not None:
        w = int(window)
        if w < 0:
            raise ValueError("functional_warping: window must be non-negative")
        if w < abs(n - m):
            w = abs(n - m)
    else:
        w = None

    def d(i, j):
        r = a[i] - b[j]
        r = r if r >= 0.0 else -r
        return r * r if cost == "sq" else r

    g = [[_INF] * m for _ in range(n)]
    for i in range(n):
        lo = 0 if w is None else max(0, i - w)
        hi = m - 1 if w is None else min(m - 1, i + w)
        for j in range(lo, hi + 1):
            if i == 0 and j == 0:
                g[i][j] = d(0, 0)
                continue
            best = _INF
            if i > 0 and g[i - 1][j] < best:
                best = g[i - 1][j]
            if j > 0 and g[i][j - 1] < best:
                best = g[i][j - 1]
            if i > 0 and j > 0 and g[i - 1][j - 1] < best:
                best = g[i - 1][j - 1]
            g[i][j] = d(i, j) + best if best < _INF else _INF
    dist = g[n - 1][m - 1]
    if dist == _INF:
        raise ValueError("functional_warping: the window is too narrow to admit any path")
    path = []
    i, j = n - 1, m - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        cand = []
        if i > 0 and j > 0:
            cand.append((g[i - 1][j - 1], i - 1, j - 1))
        if i > 0:
            cand.append((g[i - 1][j], i - 1, j))
        if j > 0:
            cand.append((g[i][j - 1], i, j - 1))
        bestv, bi, bj = cand[0]
        for c in cand[1:]:
            if c[0] < bestv:
                bestv, bi, bj = c
        i, j = bi, bj
    path.reverse()
    L = len(path)
    return RichResult(
        title="Dynamic time warping",
        summary_lines=[("len x", n), ("len y", m), ("distance", dist), ("path length", L)],
        payload={
            "estimate": dist,
            "distance": dist,
            "normalized": dist / float(L),
            "path_length": L,
            "path": [[float(p[0]), float(p[1])] for p in path],
            "n": n,
            "m": m,
            "method": "Sakoe-Chiba (1978) symmetric DP recursion g(i,j) = d(i,j) + min[g(i-1,j), g(i-1,j-1), g(i,j-1)]",
        },
    )


def cheatsheet():
    return "fdwarp: dynamic time warping alignment"
