# morie.fn -- function file (rootcoder007/morie)
"""Deep Graph Infomax objective."""

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["dgi"]


def dgi(G, X, encoder=None, seed=42):
    """
    Deep Graph Infomax

    Formula: max MI(local h_v, global s)

    One propagation step gives node summaries h = sigma(D^-1 A X W); the
    graph summary s is their mean passed through a sigmoid; the
    discriminator D(h, s) = sigmoid(h' M s) is trained to separate real
    node summaries from those of a corrupted graph whose feature rows
    are shuffled.  The objective is the binary cross-entropy of that
    discriminator, so at M = 0 every score is 1/2 and the loss is
    exactly log 2 -- the value that pins the sign convention.

    Parameters
    ----------
    G : array-like
        n x n adjacency matrix.
    X : array-like
        n x f node feature matrix.
    encoder : array-like or None
        f x d encoder weights; None draws them from the deterministic
        stream.
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (loss), loss, h, s, pos_score, neg_score, n, d.

    References
    ----------
    Velickovic, Fedus, Hamilton, Lio, Bengio & Hjelm (2019), Deep Graph
    Infomax, ICLR 2019.
    """
    A = core.mat(G)
    n = len(A)
    if n == 0:
        raise ValueError("empty input: G has no rows")
    if any(len(r) != n for r in A):
        raise ValueError("G must be a square adjacency matrix")
    Xm = core.mat(X)
    if len(Xm) != n:
        raise ValueError("X must have one row per node")
    f = len(Xm[0])
    rng = np.random.default_rng(seed)
    if encoder is None:
        d = min(f, 4)
        Wm = [[float(rng.normal(0.0, 1.0)) / math.sqrt(f) for _ in range(d)]
              for _ in range(f)]
        M = [[0.0] * d for _ in range(d)]
    else:
        Wm = core.mat(encoder)
        if len(Wm) != f:
            raise ValueError("encoder must have one row per feature")
        d = len(Wm[0])
        M = [[1.0 if a == b else 0.0 for b in range(d)] for a in range(d)]

    def propagate(F):
        H = []
        for i in range(n):
            deg = sum(A[i]) or 1.0
            agg = [sum(A[i][j] * F[j][t] for j in range(n)) / deg
                   for t in range(f)]
            H.append([core.sigmoid(sum(agg[t] * Wm[t][c] for t in range(f)))
                      for c in range(d)])
        return H

    H = propagate(Xm)
    perm = [(i * 7 + 3) % n for i in range(n)]
    seen = []
    for v in perm:
        if v not in seen:
            seen.append(v)
    for i in range(n):
        if i not in seen:
            seen.append(i)
    Hc = propagate([Xm[seen[i]] for i in range(n)])
    s = [core.sigmoid(sum(H[i][c] for i in range(n)) / n) for c in range(d)]
    pos, neg = [], []
    for i in range(n):
        a = sum(H[i][c] * sum(M[c][b] * s[b] for b in range(d))
                for c in range(d))
        b = sum(Hc[i][c] * sum(M[c][b2] * s[b2] for b2 in range(d))
                for c in range(d))
        pos.append(core.sigmoid(a))
        neg.append(core.sigmoid(b))
    loss = -(sum(math.log(v + 1e-300) for v in pos)
             + sum(math.log(1.0 - v + 1e-300) for v in neg)) / (2.0 * n)
    return RichResult(payload={
        "estimate": loss,
        "loss": loss,
        "h": H,
        "s": s,
        "pos_score": pos,
        "neg_score": neg,
        "n": n,
        "d": d,
        "method": "Deep Graph Infomax objective",
    })


def cheatsheet():
    return "dgi: Deep Graph Infomax objective"
