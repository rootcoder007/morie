# morie.fn -- function file (rootcoder007/morie)
"""DiffPool differentiable graph pooling."""

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["diffpool"]


def diffpool(A, X, K_clusters=2, S=None, seed=42):
    """
    DiffPool differentiable graph pooling

    Formula: S = softmax(GNN_pool); H' = S^T H

    The assignment matrix S is soft and learned, so the coarsened graph
    A' = S' A S and features H' = S' H are differentiable in it.  Every
    row of S sums to one by construction; when S is one-hot, A'_rs is
    exactly the number of edges between cluster r and cluster s, which
    is the hard-assignment case this reduces to.  The auxiliary link
    prediction loss is ||A - S S'||_F / n.

    Parameters
    ----------
    A : array-like
        n x n adjacency matrix.
    X : array-like
        n x f node features.
    K_clusters : int
        Number of clusters to pool into.
    S : array-like or None
        n x K assignment logits; None draws them from the deterministic
        stream.
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (link loss), S, A_pool, H_pool, link_loss,
        entropy_loss, n, K.

    References
    ----------
    Ying, You, Morris, Ren, Hamilton & Leskovec (2018), Hierarchical
    Graph Representation Learning with Differentiable Pooling,
    NeurIPS 31:4800-4810.
    """
    Am = core.mat(A)
    n = len(Am)
    if n == 0:
        raise ValueError("empty input: A has no rows")
    if any(len(r) != n for r in Am):
        raise ValueError("A must be a square adjacency matrix")
    Xm = core.mat(X)
    if len(Xm) != n:
        raise ValueError("X must have one row per node")
    f = len(Xm[0])
    K = int(K_clusters)
    if K < 1:
        raise ValueError("K_clusters must be at least 1")
    rng = np.random.default_rng(seed)
    if S is None:
        logit = [[float(rng.normal(0.0, 1.0)) for _ in range(K)]
                 for _ in range(n)]
    else:
        logit = core.mat(S)
        if len(logit) != n or len(logit[0]) != K:
            raise ValueError("S must be an n x K matrix")
    Sm = [core.softmax(r) for r in logit]
    Ap = [[0.0] * K for _ in range(K)]
    for r in range(K):
        for s in range(K):
            acc = 0.0
            for i in range(n):
                for j in range(n):
                    acc += Sm[i][r] * Am[i][j] * Sm[j][s]
            Ap[r][s] = acc
    Hp = [[sum(Sm[i][r] * Xm[i][t] for i in range(n)) for t in range(f)]
          for r in range(K)]
    ll = 0.0
    for i in range(n):
        for j in range(n):
            ll += (Am[i][j] - sum(Sm[i][r] * Sm[j][r] for r in range(K))) ** 2
    ll = math.sqrt(ll) / n
    ent = -sum(sum(v * math.log(v + 1e-300) for v in Sm[i])
               for i in range(n)) / n
    return RichResult(payload={
        "estimate": ll,
        "S": Sm,
        "A_pool": Ap,
        "H_pool": Hp,
        "link_loss": ll,
        "entropy_loss": ent,
        "n": n,
        "K": K,
        "method": "DiffPool differentiable graph pooling",
    })


def cheatsheet():
    return "diffP: DiffPool differentiable graph pooling"
