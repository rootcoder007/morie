# morie.fn -- function file (rootcoder007/morie)
"""DeepWalk: uniform random walks plus skip-gram embeddings."""

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["deepwalk"]


def adjacency_lists(G):
    """Neighbour lists from a square 0/1 adjacency matrix."""
    A = core.mat(G)
    n = len(A)
    if n == 0:
        raise ValueError("empty input: G has no rows")
    if any(len(r) != n for r in A):
        raise ValueError("G must be a square adjacency matrix")
    nb = [[j for j in range(n) if j != i and A[i][j] != 0.0] for i in range(n)]
    return A, n, nb


def uniform_walk(nb, start, length, rng):
    """One uniform random walk: every neighbour has probability 1/deg."""
    w = [start]
    cur = start
    for _ in range(length - 1):
        d = len(nb[cur])
        if d == 0:
            break
        u = float(rng.uniform(0.0, 1.0))
        k = int(u * d)
        if k >= d:
            k = d - 1
        cur = nb[cur][k]
        w.append(cur)
    return w


def skipgram(walks, n, dim, window, epochs, lr, neg, rng):
    """Skip-gram with negative sampling, plain SGD, fixed schedule."""
    W = [[float(rng.uniform(-0.5, 0.5)) / dim for _ in range(dim)]
         for _ in range(n)]
    C = [[0.0] * dim for _ in range(n)]
    for _ in range(epochs):
        for w in walks:
            for i in range(len(w)):
                for j in range(max(0, i - window), min(len(w), i + window + 1)):
                    if i == j:
                        continue
                    tgt = w[i]
                    ctx = w[j]
                    pairs = [(ctx, 1.0)]
                    for _ in range(neg):
                        k = int(float(rng.uniform(0.0, 1.0)) * n)
                        if k >= n:
                            k = n - 1
                        pairs.append((k, 0.0))
                    for (c, lab) in pairs:
                        s = sum(W[tgt][d] * C[c][d] for d in range(dim))
                        g = (core.sigmoid(s) - lab) * lr
                        for d in range(dim):
                            wt = W[tgt][d]
                            W[tgt][d] = wt - g * C[c][d]
                            C[c][d] = C[c][d] - g * wt
    return W, C


def deepwalk(G, walk_len=10, dim=8, n_walks=4, window=3, epochs=1, lr=0.05,
             neg=2, seed=42):
    """
    DeepWalk node embeddings

    Formula: random walks + skip-gram

    Truncated uniform random walks are treated as sentences and fed to
    skip-gram with negative sampling.  The walk itself is the whole
    model assumption: from a node of degree d each neighbour is taken
    with probability exactly 1/d, so the stationary distribution is
    proportional to degree.

    Parameters
    ----------
    G : array-like
        n x n adjacency matrix.
    walk_len : int
        Length of each walk.
    dim : int
        Embedding dimension.
    n_walks : int
        Walks started from each node.
    window : int
        Skip-gram context window.
    epochs : int
        Passes over the corpus.
    lr : float
        SGD step size.
    neg : int
        Negative samples per positive pair.
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (mean within-neighbour cosine), embedding,
        walks, n_walks_total, degree, n, dim.

    References
    ----------
    Perozzi, Al-Rfou & Skiena (2014), DeepWalk: Online Learning of
    Social Representations, KDD 2014:701-710.
    """
    A, n, nb = adjacency_lists(G)
    walk_len = int(walk_len)
    dim = int(dim)
    if walk_len < 2:
        raise ValueError("walk_len must be at least 2")
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if int(n_walks) < 1:
        raise ValueError("n_walks must be at least 1")
    rng = np.random.default_rng(seed)
    walks = []
    for _ in range(int(n_walks)):
        for v in range(n):
            walks.append(uniform_walk(nb, v, walk_len, rng))
    W, _C = skipgram(walks, n, dim, int(window), int(epochs), float(lr),
                     int(neg), rng)
    tot = 0.0
    cnt = 0
    for i in range(n):
        for j in nb[i]:
            a = math.sqrt(sum(v * v for v in W[i]))
            b = math.sqrt(sum(v * v for v in W[j]))
            if a > 0.0 and b > 0.0:
                tot += sum(W[i][d] * W[j][d] for d in range(dim)) / (a * b)
                cnt += 1
    return RichResult(payload={
        "estimate": tot / cnt if cnt else float("nan"),
        "embedding": W,
        "walks": walks,
        "n_walks_total": len(walks),
        "degree": [len(nb[i]) for i in range(n)],
        "n": n,
        "dim": dim,
        "method": "DeepWalk node embeddings",
    })


def cheatsheet():
    return "deepw: DeepWalk node embeddings"
