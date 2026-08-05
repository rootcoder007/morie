# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Stochastic Kronecker graph edge-probability model.

Leskovec, Chakrabarti, Kleinberg, Faloutsos and Ghahramani (2010),
"Kronecker graphs: an approach to modeling networks", Journal of
Machine Learning Research 11:985-1042.  The model takes an n0 x n0
initiator matrix Theta of edge probabilities and forms its k-th
Kronecker power

    P = Theta^[k],   P[i, j] = prod_{l=0}^{k-1} Theta[i_l, j_l],

where i_l, j_l are the base-n0 digits of i and j.  P is the matrix of
independent edge probabilities on N = n0^k nodes, so the whole
distribution is determined by Theta and k and nothing has to be
sampled: the expected edge count is the sum of P, which by the
multiplicativity of the Kronecker product equals (sum Theta)^k exactly.

Only the summary functionals and, for small graphs, the probability
matrix itself are returned; drawing a realisation is a separate step
and is deliberately not folded in here, since it would make a closed
form stochastic for no reason.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["kronecker_graph"]

_MAX_DENSE = 64


def kronecker_graph(seed, k):
    """Edge-probability matrix of a stochastic Kronecker graph.

    Parameters
    ----------
    seed : array-like
        Square initiator matrix Theta with entries in [0, 1].
    k : int
        Kronecker power; the graph has n0^k nodes.
    """
    T = core.mat(seed)
    n0 = len(T)
    if n0 == 0:
        raise ValueError("kronecker_graph: seed is empty")
    for row in T:
        if len(row) != n0:
            raise ValueError("kronecker_graph: seed must be square")
        for v in row:
            if v < 0.0 or v > 1.0:
                raise ValueError("kronecker_graph: seed entries must lie in [0, 1]")
    kk = int(k)
    if kk < 1:
        raise ValueError("kronecker_graph: k must be at least 1")
    N = n0 ** kk
    if N > 4096:
        raise ValueError("kronecker_graph: n0^k exceeds 4096 nodes")
    P = [[1.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            a = i
            b = j
            v = 1.0
            for _ in range(kk):
                v *= T[a % n0][b % n0]
                a //= n0
                b //= n0
            P[i][j] = v
    tot = 0.0
    diag = 0.0
    mn = 1.0
    mx = 0.0
    for i in range(N):
        for j in range(N):
            v = P[i][j]
            tot += v
            if i == j:
                diag += v
            if v < mn:
                mn = v
            if v > mx:
                mx = v
    deg = [sum(P[i]) for i in range(N)]
    flat = []
    if N <= _MAX_DENSE:
        for i in range(N):
            flat.extend(P[i])
    return RichResult(
        title="Stochastic Kronecker graph",
        summary_lines=[("nodes", N), ("expected edges", tot), ("k", kk)],
        payload={
            "estimate": tot,
            "expected_edges": tot,
            "expected_self_loops": diag,
            "n_nodes": float(N),
            "mean_degree": tot / N,
            "max_degree": max(deg),
            "min_degree": min(deg),
            "p_min": mn,
            "p_max": mx,
            "density": tot / (N * N),
            "seed_sum": sum(sum(row) for row in T),
            "k": float(kk),
            "P": flat,
            "n": N,
            "method": "P = Theta^[k]; sum(P) = (sum Theta)^k, Leskovec et al (2010) JMLR 11:985-1042",
        },
    )


def cheatsheet():
    return "krfgrp: Stochastic Kronecker graph"


# compact alias per ledger/NAMING.md
kroneckergraph = kronecker_graph
