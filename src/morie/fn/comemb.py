# morie.fn -- function file (rootcoder007/morie)
"""node2vec: second-order biased random walks plus skip-gram."""

import math

from . import _array_core as np
from ._richresult import RichResult
from .deepw import adjacency_lists, skipgram

__all__ = ["node2vec"]


def n2v_probs(nb, prev, cur, p, q):
    """Unnormalised second-order weights out of cur, having come from prev.

    1/p to step back, 1 to a neighbour of prev, 1/q otherwise.  At
    p = q = 1 every weight is 1, so the walk is exactly DeepWalk's
    uniform walk -- the reduction that pins this function.
    """
    out = []
    for x in nb[cur]:
        if prev is None:
            w = 1.0
        elif x == prev:
            w = 1.0 / p
        elif x in nb[prev]:
            w = 1.0
        else:
            w = 1.0 / q
        out.append(w)
    return out


def node2vec(G, p=1.0, q=1.0, dim=8, walk_len=10, n_walks=4, window=3,
             epochs=1, lr=0.05, neg=2, seed=42):
    """
    node2vec embeddings

    Formula: biased random walks + skip-gram

    The second-order walk interpolates between breadth-first and
    depth-first exploration: the return parameter p penalises stepping
    straight back, the in-out parameter q penalises leaving the
    neighbourhood of the previous node.  At p = q = 1 all weights are
    equal and the walk degenerates to DeepWalk.

    Parameters
    ----------
    G : array-like
        n x n adjacency matrix.
    p : float
        Return parameter, strictly positive.
    q : float
        In-out parameter, strictly positive.
    dim : int
        Embedding dimension.
    walk_len, n_walks, window, epochs, lr, neg : see deepw.
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (mean within-neighbour cosine), embedding,
        walks, degree, n, dim, p, q.

    References
    ----------
    Grover & Leskovec (2016), node2vec: Scalable Feature Learning for
    Networks, KDD 2016:855-864.
    """
    A, n, nb = adjacency_lists(G)
    if not (p > 0.0 and q > 0.0):
        raise ValueError("p and q must be strictly positive")
    dim = int(dim)
    walk_len = int(walk_len)
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if walk_len < 2:
        raise ValueError("walk_len must be at least 2")
    rng = np.random.default_rng(seed)
    walks = []
    for _ in range(int(n_walks)):
        for v in range(n):
            w = [v]
            prev = None
            cur = v
            for _ in range(walk_len - 1):
                if not nb[cur]:
                    break
                wt = n2v_probs(nb, prev, cur, p, q)
                tot = sum(wt)
                u = float(rng.uniform(0.0, 1.0)) * tot
                acc = 0.0
                pick = len(wt) - 1
                for k in range(len(wt)):
                    acc += wt[k]
                    if u <= acc:
                        pick = k
                        break
                prev, cur = cur, nb[cur][pick]
                w.append(cur)
            walks.append(w)
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
        "degree": [len(nb[i]) for i in range(n)],
        "n": n,
        "dim": dim,
        "p": p,
        "q": q,
        "method": "node2vec biased-walk embeddings",
    })


def cheatsheet():
    return "comemb: node2vec biased-walk embeddings"
