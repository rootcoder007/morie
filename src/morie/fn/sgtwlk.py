# morie.fn -- function file (rootcoder007/morie)
"""Weisfeiler-Leman colour refinement."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sgt_weisfeiler_leman_relabel"]


def sgt_weisfeiler_leman_relabel(A, labels0=None, max_iter=3):
    """Refine node colours until neighbourhoods stop distinguishing them.

    The test is the standard cheap check for graph non-isomorphism: if
    two graphs end with different colour multisets they are certainly
    different.  The converse fails -- regular graphs of the same degree
    are indistinguishable to it -- and that failure is precisely the
    known expressiveness ceiling of message-passing graph networks,
    which is why the algorithm keeps appearing in that literature.

    Determinism: each round hashes the sorted multiset of neighbour
    colours to a canonical string and assigns new integer colours in
    order of first appearance, so no hash-seed randomness enters.

    Formula: ``h^(t+1)(v) = hash(h^t(v), {{h^t(u) : u in N(v)}})``.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Adjacency.
    labels0 : array-like, optional
        Starting colours; all-zero by default.
    max_iter : int, default 3
        Refinement rounds.

    Returns
    -------
    RichResult
        ``labels_t``, ``estimate`` (number of distinct final colours),
        ``history`` (colour counts per round), ``n``.

    References
    ----------
    Weisfeiler, B. & Leman, A. A. (1968).  The reduction of a graph to
    canonical form and the algebra which appears therein.
    Nauchno-Tekhnicheskaya Informatsia 2(9):12-16.  The
    graph-neural-network connection is Xu, K., Hu, W., Leskovec, J. &
    Jegelka, S. (2019), How powerful are graph neural networks?, ICLR
    2019.
    """
    Am = C.mat(A)
    n = len(Am)
    lab = [int(round(v)) for v in C.vec(labels0)] if labels0 is not None else [0] * n
    hist = [len(set(lab))]
    for _ in range(int(max_iter)):
        keys = []
        for v in range(n):
            nb = sorted(lab[u] for u in range(n) if u != v and Am[v][u] != 0.0)
            keys.append(str(lab[v]) + "|" + ",".join(str(t) for t in nb))
        order = []
        for kk in keys:
            if kk not in order:
                order.append(kk)
        lab = [order.index(kk) for kk in keys]
        hist.append(len(order))
    return RichResult(payload={
        "labels_t": lab, "estimate": float(len(set(lab))), "history": hist,
        "n": n, "method": "Weisfeiler-Leman colour refinement"})


def cheatsheet():
    return "sgtwlk: Weisfeiler-Leman colour refinement."
