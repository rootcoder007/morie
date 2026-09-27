# morie.fn -- function file (rootcoder007/morie)
"""Spatial correlogram over neighbour-graph lag orders (Moran's I, Geary's C, correlation)."""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats
from ._richresult import RichResult

__all__ = ["spatial_correlogram", "graph_lags"]


def graph_lags(adjacency, order):
    """Neighbours at exactly each graph distance ``1..order``.

    Unit ``j`` is a lag-``k`` neighbour of ``i`` when the shortest path
    from ``i`` to ``j`` in the neighbour graph has ``k`` edges, as
    ``spdep::nblag``.  Returns ``order`` binary (n, n) matrices as lists.
    """
    A = np.asarray(adjacency, dtype=float).tolist()
    n = len(A)
    nbr = [[j for j in range(n) if j != i and A[i][j] != 0.0] for i in range(n)]
    lags = [[[0.0] * n for _ in range(n)] for _ in range(order)]
    for i in range(n):
        dist = {i: 0}
        frontier = [i]
        for k in range(1, order + 1):
            nxt = []
            for u in frontier:
                for v in nbr[u]:
                    if v not in dist:
                        dist[v] = k
                        nxt.append(v)
                        lags[k - 1][i][v] = 1.0
            frontier = nxt
    return lags


def _style(B, style):
    n = len(B)
    rs = [sum(r) for r in B]
    s0 = sum(rs)
    eff = sum(1 for v in rs if v > 0.0)
    if style == "B":
        return [list(r) for r in B]
    if style == "W":
        return [[v / rs[i] if rs[i] > 0.0 else 0.0 for v in B[i]] for i in range(n)]
    if style == "C":
        return [[v * eff / s0 for v in r] for r in B]
    if style == "U":
        return [[v / s0 for v in r] for r in B]
    raise ValueError("style must be one of B, W, C, U")


def _constants(W):
    n = len(W)
    rs = [sum(r) for r in W]
    cs = [sum(W[i][j] for i in range(n)) for j in range(n)]
    s0 = sum(rs)
    s1 = 0.5 * sum((W[i][j] + W[j][i]) ** 2 for i in range(n) for j in range(n))
    s2 = sum((rs[i] + cs[i]) ** 2 for i in range(n))
    m = float(sum(1 for i in range(n) if any(W[i][j] != 0.0 for j in range(n))))
    return m, s0, s1, s2


def _moran(x, W, randomisation):
    N = len(x)
    n, s0, s1, s2 = _constants(W)
    mu = sum(x) / N
    z = [v - mu for v in x]
    zz = sum(v * v for v in z)
    K = N * sum(v**4 for v in z) / zz**2
    lz = [sum(W[i][j] * z[j] for j in range(N)) for i in range(N)]
    stat = (n / s0) * sum(z[i] * lz[i] for i in range(N)) / zz
    e = -1.0 / (n - 1.0)
    nn, sq = n * n, s0 * s0
    if randomisation:
        v = n * (s1 * (nn - 3.0 * n + 3.0) - n * s2 + 3.0 * sq)
        v -= K * (s1 * (nn - n) - 2.0 * n * s2 + 6.0 * sq)
        v = v / ((n - 1.0) * (n - 2.0) * (n - 3.0) * sq) - e * e
    else:
        v = (nn * s1 - n * s2 + 3.0 * sq) / (sq * (nn - 1.0)) - e * e
    return stat, e, v


def _geary(x, W, randomisation):
    N = len(x)
    n, s0, s1, s2 = _constants(W)
    mu = sum(x) / N
    sd = (sum((v - mu) ** 2 for v in x) / (N - 1)) ** 0.5
    z = [(v - mu) / sd for v in x]
    zz = sum(v * v for v in z)
    K = n * sum(v**4 for v in z) / zz**2
    num = sum(W[i][j] * (z[i] - z[j]) ** 2 for i in range(N) for j in range(N) if j != i)
    n1 = n - 1.0
    stat = (n1 / (2.0 * s0)) * num / zz
    nn, sq = n * n, s0 * s0
    if randomisation:
        v = n1 * s1 * (nn - 3.0 * N + 3.0 - K * n1)
        v -= 0.25 * (n1 * s2 * (nn + 3.0 * N - 6.0 - K * (nn - N + 2.0)))
        v += sq * (nn - 3.0 - K * n1 * n1)
        v /= N * (n - 2.0) * (n - 3.0) * sq
    else:
        v = ((2.0 * s1 + s2) * n1 - 4.0 * sq) / (2.0 * (N + 1.0) * sq)
    return stat, 1.0, v


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    saa = sum((v - ma) ** 2 for v in a)
    sbb = sum((v - mb) ** 2 for v in b)
    return sab / (saa * sbb) ** 0.5


def spatial_correlogram(
    adjacency,
    x,
    order: int = 1,
    method: str = "I",
    style: str = "W",
    randomisation: bool = True,
) -> RichResult:
    r"""Spatial correlogram along the lag orders of a neighbour graph.

    For each lag ``k = 1..order`` the lag-``k`` neighbours (shortest-path
    distance exactly ``k``, :func:`graph_lags`) are coded with ``style``
    and a global statistic is computed, as ``spdep::sp.correlogram``:

    - ``method="I"``: Moran's I with its expectation ``-1/(n-1)`` and its
      variance under randomisation or normality (Cliff and Ord 1981), as
      ``spdep::moran.test``;
    - ``method="C"``: Geary's C with expectation 1 and variance under
      randomisation or normality, as ``spdep::geary.test``;
    - ``method="corr"``: the Pearson correlation of ``x`` with its lag-``k``
      spatial lag.

    ``n`` in the moments counts the units with at least one lag-``k``
    neighbour.  The deviate is ``(estimate - expectation) / sqrt(variance)``
    with a two-sided normal p-value.

    :param adjacency: (n, n) neighbour indicator (nonzero = neighbour).
    :param x: Values (n,).
    :param order: Largest lag order.
    :param method: ``I``, ``C`` or ``corr``.
    :param style: Weight coding ``W``, ``B``, ``C`` or ``U``.
    :param randomisation: Randomisation (else normality) variance.
    :return: :class:`RichResult` with ``estimate``, ``expectation``,
        ``variance``, ``z``, ``p_value`` (lists by lag; only ``estimate``
        for ``corr``), ``n_with_neighbours``, ``method``.
    :raises ValueError: If a lag has fewer than three units with neighbours.

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and
    Applications*. Pion, London.

    Examples
    --------
    >>> A = [[1 if abs(i - j) == 1 else 0 for j in range(8)] for i in range(8)]
    >>> x = [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 8.0, 7.0]
    >>> [round(v, 6) for v in spatial_correlogram(A, x, order=2).estimate]
    [0.797619, 0.25]
    """
    xs = [float(v) for v in np.asarray(x, dtype=float).tolist()]
    A = np.asarray(adjacency, dtype=float).tolist()
    N = len(xs)
    if len(A) != N or any(len(r) != N for r in A):
        raise ValueError("adjacency must be n x n with n = len(x)")
    if int(order) < 1:
        raise ValueError("order must be at least 1")
    if method not in ("I", "C", "corr"):
        raise ValueError("method must be I, C or corr")
    lags = graph_lags(A, int(order))
    est, ex, va, zs, ps, counts = [], [], [], [], [], []
    for k, B in enumerate(lags, start=1):
        cnt = sum(1 for r in B if any(v != 0.0 for v in r))
        if cnt < 3:
            raise ValueError(f"too few units with lag-{k} neighbours; reduce order")
        counts.append(cnt)
        W = _style(B, style)
        if method == "corr":
            lag = [sum(W[i][j] * xs[j] for j in range(N)) for i in range(N)]
            est.append(_pearson(xs, lag))
            continue
        s, e, v = (_moran if method == "I" else _geary)(xs, W, randomisation)
        z = (s - e) / v**0.5
        est.append(s)
        ex.append(e)
        va.append(v)
        zs.append(z)
        ps.append(float(2.0 * stats.norm.sf(abs(z))))
    return RichResult(
        payload={
            "estimate": est,
            "expectation": ex,
            "variance": va,
            "z": zs,
            "p_value": ps,
            "n_with_neighbours": counts,
            "method": method,
            "style": style,
        }
    )


def cheatsheet() -> str:
    return "spatial_correlogram(A, x, order, method) -> Moran/Geary/correlation by graph lag (spdep::sp.correlogram)."
