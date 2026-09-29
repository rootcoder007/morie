# morie.fn -- function file (rootcoder007/morie)
"""Community detection by the map equation (Infomap): the two-level description length of a random
walk on an undirected weighted network under a module partition, and its minimisation by
deterministic local moving of nodes and of aggregated modules (the Infomap core algorithm)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = ["map_equation", "infomap_partition"]


def _plogp(p):
    return p * math.log2(p) if p > 0 else 0.0


def _adj(A):
    M = [[float(v) for v in r] for r in np.asarray(A, dtype=float).tolist()]
    n = len(M)
    return [[(M[i][j] + M[j][i]) / 2.0 for j in range(n)] for i in range(n)]


def _flow(A, flow):
    n = len(A)
    s = [ssum(r) for r in A]
    if flow == "undirected":
        tot = ssum(s)
        return [v / tot for v in s], [[A[i][j] / tot for j in range(n)] for i in range(n)], None
    if flow != "igraph":
        raise ValueError("flow must be 'undirected' or 'igraph'")
    # igraph's FlowGraph: PageRank with uniform teleportation alpha = 0.15, its stopping rules
    alpha = 0.15
    beta = 1 - alpha
    P = [[A[i][j] / s[i] if s[i] > 0 else 0.0 for j in range(n)] for i in range(n)]
    dang = [i for i in range(n) if s[i] <= 0]
    tmp = [1.0 / n] * n
    sq, it = 1.0, 0
    while True:
        dsz = ssum(tmp[i] for i in dang)
        size = [(alpha + beta * dsz) / n] * n
        for i in range(n):
            for j in range(n):
                if P[i][j]:
                    size[j] += beta * P[i][j] * tmp[i]
        tot = ssum(size)
        size = [v / tot for v in size]
        old = sq
        sq = ssum(abs(a - b) for a, b in zip(size, tmp))
        tmp = size
        it += 1
        if sq == old:
            alpha += 1.0e-10
            beta = 1.0 - alpha
        if not (it < 200 and (sq > 1.0e-15 or it < 50)):
            break
    F = [[beta * tmp[i] * P[i][j] for j in range(n)] for i in range(n)]
    d = [tmp[i] if s[i] <= 0 else 0.0 for i in range(n)]
    return tmp, F, (alpha, beta, d)


def _codelength(p, F, tele, member):
    n = len(p)
    mods = sorted(set(member))
    idx = {m: [i for i in range(n) if member[i] == m] for m in mods}
    ex, pin = {}, {}
    for m in mods:
        pm = ssum(p[i] for i in idx[m])
        pin[m] = pm
        if tele is None:
            ex[m] = ssum(F[i][j] for i in idx[m] for j in range(n) if member[j] != m)
        else:
            alpha, beta, d = tele
            dm = ssum(d[i] for i in idx[m])
            internal = ssum(F[i][j] for i in idx[m] for j in idx[m])
            ex[m] = pm - (alpha * pm + beta * dm) * len(idx[m]) / n - internal
    q = ssum(ex[m] for m in mods)
    return (
        _plogp(q)
        - 2 * ssum(_plogp(ex[m]) for m in mods)
        - ssum(_plogp(v) for v in p)
        + ssum(_plogp(ex[m] + pin[m]) for m in mods)
    )


def map_equation(A, membership, flow="undirected"):
    r"""Two-level map equation ``L(M)`` (bits per step) of a partition of an undirected network.

    With node visit rates ``p_a = s_a / 2W`` (strength over twice the total
    weight) and module exit rates ``q_i = sum_{a in i, b not in i} w_ab / 2W``,
    ``L = q log q - 2 sum_i q_i log q_i - sum_a p_a log p_a + sum_i (q_i + p_i)
    log(q_i + p_i)`` with ``q = sum_i q_i`` and ``p_i = sum_{a in i} p_a``
    (logarithms base 2). The one-module partition gives the entropy of the
    visit rates. ``flow="igraph"`` uses the flow model of ``igraph::cluster_infomap``
    instead: PageRank visit rates with uniform teleportation 0.15, link flows
    ``0.85 p_a w_ab / s_a`` and module exit rates that include teleportation out
    of the module.

    References
    ----------
    Rosvall, M. and Bergstrom, C. T. (2008). Maps of random walks on complex
    networks reveal community structure. *PNAS* 105, 1118-1123.

    Examples
    --------
    >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
    ...      [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
    >>> round(map_equation(A, [0, 0, 0, 1, 1, 1]), 12)
    2.320730356834
    """
    p, F, tele = _flow(_adj(A), flow)
    return _codelength(p, F, tele, list(membership))


def infomap_partition(A, flow="undirected", max_passes=100):
    r"""Two-level Infomap: minimise the map equation by local moving and aggregation.

    Starting from singletons, nodes are visited in index order and moved to
    the neighbouring module (or kept) that lowers ``L`` most, repeating until a
    full pass brings no improvement; the modules are then aggregated and the
    same moves are made with whole modules, until nothing changes. Moves use
    the exact map equation, ties keep the current module, and the procedure
    is deterministic. Modules are relabelled 0, 1, ... by first node;
    ``flow`` as in :func:`map_equation`.

    References
    ----------
    Rosvall, M., Axelsson, D. and Bergstrom, C. T. (2009). The map equation.
    *European Physical Journal Special Topics* 178, 13-23.

    Examples
    --------
    >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
    ...      [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
    >>> infomap_partition(A).membership
    [0, 0, 0, 1, 1, 1]
    """
    M = _adj(A)
    n = len(M)
    p, F, tele = _flow(M, flow)
    member = list(range(n))
    groups = [[i] for i in range(n)]
    L = _codelength(p, F, tele, member)
    for _ in range(max_passes):
        moved_any = False
        improved = True
        while improved:
            improved = False
            for g in groups:
                cur = member[g[0]]
                nb = sorted({member[j] for i in g for j in range(n) if M[i][j] and member[j] != cur})
                best, best_m = L, cur
                for m in nb:
                    trial = list(member)
                    for i in g:
                        trial[i] = m
                    Lt = _codelength(p, F, tele, trial)
                    if Lt < best - 1e-12:
                        best, best_m = Lt, m
                if best_m != cur:
                    for i in g:
                        member[i] = best_m
                    L = best
                    improved = moved_any = True
        labels = sorted(set(member))
        new_groups = [[i for i in range(n) if member[i] == m] for m in labels]
        if not moved_any or len(new_groups) == len(groups):
            groups = new_groups
            break
        groups = new_groups
    first = {}
    out = []
    for m in member:
        if m not in first:
            first[m] = len(first)
        out.append(first[m])
    return RichResult(payload={"membership": out, "codelength": L, "n_modules": len(first)})


def cheatsheet() -> str:
    return "map_equation / infomap_partition -> community detection by the map equation."
