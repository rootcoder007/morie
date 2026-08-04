# morie.fn -- function file (rootcoder007/morie)
"""Error and attack tolerance of a network."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['netattack', 'network_attack_tolerance']


def netattack(A, strategy="attack", k=1, seed=1):
    """Error and attack tolerance of a network.

    Scale-free networks survive random node loss and collapse under targeted removal of their hubs; the point of the measurement is the contrast between the two removal rules, not either one alone. 'error' removes nodes drawn from the shared minstd stream, 'attack' removes them in descending degree with ties broken by index. The three reported quantities are the paper's own: the diameter is the average length of the shortest paths between nodes, S is the size of the largest cluster as a fraction of the system, and <s> is the average size of the isolated fragments.


    Formula: remove f N nodes, then report S = |largest cluster| / N, <s> = mean size of the remaining fragments, and d = mean shortest path

    Parameters
    ----------
    A : array-like, shape (n, n)
        Symmetric 0/1 adjacency matrix.
    strategy : {'attack', 'error'}
        Targeted (highest degree first) or random removal.
    k : int
        Number of nodes removed.
    seed : int
        Seed of the shared minstd stream used by 'error'.

    Returns
    -------
    RichResult
        ``s_giant``, ``mean_fragment``, ``diameter``, ``removed``, ``n_components``, ``n``.

    References
    ----------
    Albert, Jeong and Barabasi (2000), Error and attack tolerance of
    complex networks, Nature 406:378-382, arXiv:cond-mat/0008064.
    Verified against the paper for the definitions of d, S and <s>.
    """
    A = C.mat(A)
    n = len(A)
    k = int(k)
    deg = [sum(1 for j in range(n) if i != j and A[i][j] != 0) for i in range(n)]
    if strategy == "attack":
        order = sorted(range(n), key=lambda i: (-deg[i], i))
        removed = sorted(order[:k])
    elif strategy == "error":
        g = C.Lcg(seed)
        pool = list(range(n)); removed = []
        for _ in range(min(k, n)):
            j = int(g.unif() * len(pool))
            if j >= len(pool):
                j = len(pool) - 1
            removed.append(pool.pop(j))
        removed = sorted(removed)
    else:
        raise ValueError("strategy must be 'attack' or 'error'")
    keep = [i for i in range(n) if i not in set(removed)]
    idx = {v: i for i, v in enumerate(keep)}
    m = len(keep)
    adj = [[] for _ in range(m)]
    for a in range(m):
        for b in range(m):
            if a != b and A[keep[a]][keep[b]] != 0:
                adj[a].append(b)
    seen = [False] * m
    comps = []
    for s in range(m):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(comp)
    sizes = sorted((len(c) for c in comps), reverse=True)
    giant = sizes[0] if sizes else 0
    rest = sizes[1:]
    tot, pairs = 0.0, 0
    for s in range(m):
        dist = [-1] * m
        dist[s] = 0
        q = [s]; h = 0
        while h < len(q):
            v = q[h]; h += 1
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
        for t in range(m):
            if t != s and dist[t] > 0:
                tot += dist[t]; pairs += 1
    return RichResult(payload={
        "s_giant": giant / n if n else float("nan"),
        "mean_fragment": (sum(rest) / len(rest)) if rest else float("nan"),
        "diameter": tot / pairs if pairs else float("nan"),
        "removed": removed, "n_components": len(comps), "n": n,
        "method": "Error and attack tolerance (Albert-Jeong-Barabasi)"})


network_attack_tolerance = netattack


def cheatsheet():
    return "netatp: Error and attack tolerance of a network."
