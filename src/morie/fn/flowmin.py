# morie.fn -- slice s03 (rootcoder007/morie)
"""Stoer-Wagner global minimum cut.

Source consulted: Stoer, M. and Wagner, F. (1997).  A simple min-cut
algorithm.  *Journal of the ACM* 44(4), 585-591.  The algorithm repeats
a "minimum cut phase": grow a set A from an arbitrary start vertex,
always adding the vertex most tightly connected to A; the last two
vertices added, s and t, give a cut-of-the-phase whose weight is the
weight of the cut separating t from the rest, and s and t are then
merged.  After n - 1 phases the lightest cut-of-the-phase is the global
minimum cut.  The JACM paper is paywalled; the phase construction and
the merge step are quoted in their standard published form.

DETERMINISM.  The paper's "arbitrary" start vertex and its tie-breaking
are fixed here: the phase always starts at the lowest surviving index
and ties in the tightness are broken by the lowest index, so the cut and
the partition reproduce exactly in both arms.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["min_cut"]


def min_cut(A):
    """Global minimum cut of an undirected weighted graph.

    Returns
    -------
    RichResult with payload:
        estimate  : the weight of the minimum cut
        weight    : same as estimate
        partition : 0/1 membership per original vertex
        phases    : the cut-of-the-phase weight at each phase
    """
    W = k.mat(A)
    n = len(W)
    w = [[W[i][j] for j in range(n)] for i in range(n)]
    groups = [[i] for i in range(n)]
    alive = list(range(n))
    best = float("inf")
    bestset = []
    phases = []
    while len(alive) > 1:
        m = len(alive)
        inA = [False] * m
        wsum = [0.0] * m
        order = []
        for _ in range(m):
            sel = -1
            for t in range(m):
                if not inA[t] and (sel < 0 or wsum[t] > wsum[sel]):
                    sel = t
            inA[sel] = True
            order.append(sel)
            for t in range(m):
                if not inA[t]:
                    wsum[t] += w[alive[sel]][alive[t]]
        t_i = order[-1]
        s_i = order[-2]
        cut = 0.0
        for t in range(m):
            if t != t_i:
                cut += w[alive[t_i]][alive[t]]
        phases.append(cut)
        if cut < best:
            best = cut
            bestset = list(groups[alive[t_i]])
        s = alive[s_i]
        tt = alive[t_i]
        for t in range(n):
            w[s][t] = w[s][t] + w[tt][t]
            w[t][s] = w[s][t]
        w[s][s] = 0.0
        groups[s] = groups[s] + groups[tt]
        alive = [x for x in alive if x != tt]
    part = [1 if i in bestset else 0 for i in range(n)]
    return RichResult(
        title="Stoer-Wagner minimum cut",
        summary_lines=[("min cut", best)],
        payload={
            "estimate": best,
            "weight": best,
            "partition": part,
            "phases": phases,
            "n": n,
            "method": "Stoer-Wagner global minimum cut (1997), deterministic start and tie-breaking",
        },
    )


def cheatsheet():
    return "flowmin: Stoer-Wagner min-cut"


mincut = min_cut
