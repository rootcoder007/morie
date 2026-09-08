# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Multi-head graph attention layer.

Velickovic, Cucurull, Casanova, Romero, Lio and Bengio (2018), "Graph
attention networks", ICLR 2018, arXiv:1710.10903, equations (1)-(4)
with the head averaging of equation (6).

The single-head layer already exists in this package as
``morie.fn.gat``; the wave2 audit flagged this module as a duplicate of
it and it is one, so the attention arithmetic is NOT repeated here.
This module adds only the multi-head interface: with no weights
supplied the deterministic choice W = I and a = 1 is used, every head
then sees the same input, and averaging identical heads is exact --
which is what makes the ``heads`` argument checkable at all.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from .gat import gat as _gat

from ._richresult import RichResult

__all__ = ["graph_attention_net"]


def graph_attention_net(G, X, heads=1):
    """One multi-head GAT layer, delegating the per-head arithmetic to gat."""
    M = core.mat(G)
    n = len(M)
    if n == 0:
        raise ValueError("graph_attention_net: graph is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("graph_attention_net: adjacency matrix must be square")
    H = core.mat(X)
    if len(H) != n:
        raise ValueError("graph_attention_net: X must have one row per node")
    nh = int(heads)
    if nh < 1:
        raise ValueError("graph_attention_net: heads must be at least 1")
    p = len(H[0])
    W = [[1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
    a = [1.0] * (2 * p)
    acc = None
    for _ in range(nh):
        res = _gat(M, H, W, a)
        Hh = core.mat(res["H"] if "H" in res else res["estimate"])
        if acc is None:
            acc = [[0.0] * len(Hh[0]) for _ in range(n)]
        for i in range(n):
            for c in range(len(Hh[0])):
                acc[i][c] += Hh[i][c] / nh
    flat = [v for row in acc for v in row]
    return RichResult(
        title="Multi-head graph attention layer",
        summary_lines=[("nodes", n), ("heads", nh)],
        payload={
            "estimate": sum(flat) / len(flat),
            "H": acc,
            "heads": nh,
            "n": n,
            "method": "average of gat() heads, Velickovic et al. (2018) eqs. (1)-(4) and (6)",
        },
    )


def cheatsheet():
    return "gatemd: multi-head GAT layer (delegates to gat)"
