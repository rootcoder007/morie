# morie.fn -- slice s03 (rootcoder007/morie)
"""Weisfeiler-Lehman subtree graph kernel.

Source consulted (FETCHED, jmlr.org PDF): Shervashidze, N., Schweitzer,
P., van Leeuwen, E. J., Mehlhorn, K. and Borgwardt, K. M. (2011).
Weisfeiler-Lehman graph kernels.  *JMLR* 12, 2539-2561.  Algorithm 1
gives one iteration of the 1-dimensional Weisfeiler-Lehman test: assign
each node the multiset of its neighbours' previous labels, sort each
multiset, prepend the node's own previous label, and compress the
resulting string to a fresh label.  The kernel, the paper's equation
(2), is

    k^(h)_WLsubtree(G, G') = < phi^(h)(G), phi^(h)(G') >

with phi^(h)(G) = ( c_0(G, sigma_01), ..., c_h(G, sigma_h|Sigma_h|) ) the
counts of every original and compressed label over iterations 0..h.  As
the paper puts it, the kernel "counts common original and compressed
labels in two graphs".

The label alphabet is shared between the two graphs, which is what makes
the inner product meaningful; compressed labels are allocated in order
of first appearance, so the run is deterministic.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["wl_kernel"]


def _adj(G):
    A = k.mat(G)
    n = len(A)
    nb = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and A[i][j] != 0.0:
                nb[i].append(j)
    return nb


def wl_kernel(G1, G2, K=3, labels1=None, labels2=None, normalize=False):
    """WL subtree kernel between two labelled graphs.

    Parameters
    ----------
    G1, G2 : 2-D array-like
        Adjacency matrices.
    K : int
        Number of WL iterations h.
    labels1, labels2 : array-like, optional
        Initial node labels; all-equal by default (unlabelled graphs).
    normalize : bool
        Divide by sqrt(k(G1,G1) k(G2,G2)).

    Returns
    -------
    RichResult with payload:
        estimate : the kernel value
        per_iter : the contribution of each iteration 0..h
        n_labels : size of the shared label alphabet
    """
    nb1 = _adj(G1)
    nb2 = _adj(G2)
    n1 = len(nb1)
    n2 = len(nb2)
    l1 = [str(x) for x in labels1] if labels1 is not None else ["0"] * n1
    l2 = [str(x) for x in labels2] if labels2 is not None else ["0"] * n2
    alpha = []

    def code(s):
        if s not in alpha:
            alpha.append(s)
        return alpha.index(s)

    l1 = [str(code(s)) for s in l1]
    l2 = [str(code(s)) for s in l2]
    total = 0.0
    per = []
    for it in range(int(K) + 1):
        c1 = {}
        c2 = {}
        for s in l1:
            c1[s] = c1.get(s, 0.0) + 1.0
        for s in l2:
            c2[s] = c2.get(s, 0.0) + 1.0
        dot = 0.0
        for s in sorted(set(list(c1) + list(c2))):
            dot += c1.get(s, 0.0) * c2.get(s, 0.0)
        per.append(dot)
        total += dot
        if it == int(K):
            break
        n1l = []
        for v in range(n1):
            ms = sorted([l1[u] for u in nb1[v]])
            n1l.append(str(code(l1[v] + "," + "|".join(ms))))
        n2l = []
        for v in range(n2):
            ms = sorted([l2[u] for u in nb2[v]])
            n2l.append(str(code(l2[v] + "," + "|".join(ms))))
        l1 = n1l
        l2 = n2l
    est = total
    if normalize:
        s1 = wl_kernel(G1, G1, K, labels1, labels1)["estimate"]
        s2 = wl_kernel(G2, G2, K, labels2, labels2)["estimate"]
        d = math.sqrt(s1 * s2)
        est = total / d if d > 0.0 else float("nan")
    return RichResult(
        title="Weisfeiler-Lehman subtree kernel",
        summary_lines=[("k", est), ("iterations", int(K))],
        payload={
            "estimate": est,
            "kernel": total,
            "per_iter": per,
            "n_labels": len(alpha),
            "method": "Weisfeiler-Lehman subtree kernel (Shervashidze et al. 2011, eq. 2)",
        },
    )


def cheatsheet():
    return "weisL: Weisfeiler-Lehman graph kernel"


wlkernel = wl_kernel
