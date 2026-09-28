# morie.fn -- function file (rootcoder007/morie)
"""Clustering algorithms for point data: affinity propagation, CLARANS, CURE, DIANA and CHAMELEON."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform
from ._s03core import jacobi

__all__ = ["affinity_propagation", "clarans", "cure_clustering", "diana", "chameleon"]


def _pts(X):
    P = [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in X]
    if not P:
        raise ValueError("X must have rows")
    return P


def _d(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += (x - y) * (x - y)
    return math.sqrt(s)


def _first_appearance(lab):
    seen = {}
    out = []
    for v in lab:
        if v not in seen:
            seen[v] = len(seen) + 1
        out.append(seen[v])
    return out


class _Uniforms:
    """Sequential Philox uniforms, drawn in blocks of 4096 on consecutive streams."""

    def __init__(self, seed):
        self.seed, self.block, self.buf, self.pos = seed, 0, [], 0

    def next(self):
        if self.pos >= len(self.buf):
            self.buf = [float(v) for v in random_uniform(4096, seed=self.seed, stream=self.block)]
            self.block += 1
            self.pos = 0
        self.pos += 1
        return self.buf[self.pos - 1]


def affinity_propagation(
    X=None, *, S=None, preference=None, damping: float = 0.9, maxits: int = 1000, convits: int = 100
):
    r"""Affinity propagation (Frey and Dueck 2007): exemplars from responsibility and availability messages.

    Similarities default to negative squared Euclidean distances and the
    preference (diagonal) to their median. Each iteration updates
    ``r(i,k) = s(i,k) - max_{k' != k} (a(i,k') + s(i,k'))`` and
    ``a(i,k) = min(0, r(k,k) + sum_{i' not in {i,k}} max(0, r(i',k)))``
    (``a(k,k) = sum_{i' != k} max(0, r(i',k))``), each damped as
    ``damping * old + (1 - damping) * new``; points with ``a(k,k) + r(k,k) > 0``
    are exemplars. It stops when the exemplar set is unchanged for
    ``convits`` iterations. As in Frey's code and ``apcluster``, each cluster's
    exemplar is then refined to the member maximising the summed similarity
    and points are reassigned to the most similar exemplar.

    References
    ----------
    Frey, B. J. and Dueck, D. (2007). Clustering by passing messages between
    data points. *Science*, 315(5814), 972-976.

    Examples
    --------
    >>> r = affinity_propagation([[0.0, 0.0], [0.2, 0.1], [5.0, 5.0], [5.1, 4.8], [0.1, 0.3]])
    >>> r.cluster, r.exemplars
    ([1, 1, 2, 2, 1], [2, 3])
    """
    if S is None:
        P = _pts(X)
        n = len(P)
        S = [[-(_d(P[i], P[k]) ** 2) for k in range(n)] for i in range(n)]
    else:
        S = [[float(v) for v in row] for row in S]
        n = len(S)
    if preference is None:
        off = sorted(S[i][k] for i in range(n) for k in range(n) if i != k)
        m = len(off)
        preference = off[m // 2] if m % 2 else 0.5 * (off[m // 2 - 1] + off[m // 2])
    for i in range(n):
        S[i][i] = float(preference)
    lam = damping
    R = [[0.0] * n for _ in range(n)]
    A = [[0.0] * n for _ in range(n)]
    hist = []
    it = 0
    converged = False
    for it in range(1, maxits + 1):
        for i in range(n):
            AS = [A[i][k] + S[i][k] for k in range(n)]
            k1 = max(range(n), key=lambda k: (AS[k], -k))
            y1 = AS[k1]
            y2 = max(AS[k] for k in range(n) if k != k1) if n > 1 else -math.inf
            for k in range(n):
                new = S[i][k] - (y2 if k == k1 else y1)
                R[i][k] = lam * R[i][k] + (1 - lam) * new
        for k in range(n):
            col = 0.0
            for i in range(n):
                col += R[i][k] if i == k else max(R[i][k], 0.0)
            for i in range(n):
                rp = R[i][k] if i == k else max(R[i][k], 0.0)
                new = col - rp
                if i != k:
                    new = min(new, 0.0)
                A[i][k] = lam * A[i][k] + (1 - lam) * new
        E = tuple(A[k][k] + R[k][k] > 0 for k in range(n))
        hist.append(E)
        if len(hist) > convits:
            hist.pop(0)
        if it >= convits and all(h == E for h in hist) and any(E):
            converged = True
            break
    ex = [k for k in range(n) if A[k][k] + R[k][k] > 0]
    if not ex:
        return RichResult(payload={"cluster": [0] * n, "exemplars": [], "iterations": it, "converged": converged})

    def assign(ex):
        c = [max(range(len(ex)), key=lambda j: (S[i][ex[j]], -j)) for i in range(n)]
        for j, e in enumerate(ex):
            c[e] = j
        return c

    c = assign(ex)
    for j in range(len(ex)):
        mem = [i for i in range(n) if c[i] == j]
        ex[j] = max(mem, key=lambda q: (ssum(S[i][q] for i in mem), -q))
    c = assign(ex)
    exemplar_of = [ex[c[i]] for i in range(n)]
    order = sorted(set(exemplar_of))
    return RichResult(
        payload={
            "cluster": [order.index(e) + 1 for e in exemplar_of],
            "exemplars": [e + 1 for e in order],
            "iterations": it,
            "converged": converged,
            "preference": float(preference),
        }
    )


def clarans(X, k: int, *, numlocal: int = 2, maxneighbor: int | None = None, seed: int = 1) -> RichResult:
    r"""CLARANS (Ng and Han 2002): randomized search for ``k`` medoids on the graph of medoid sets.

    ``numlocal`` restarts each begin at a random medoid set and move to a
    random neighbour (one medoid swapped for one non-medoid) whenever it
    lowers the total distance to the nearest medoid; after ``maxneighbor``
    consecutive failures the node is a local minimum. The best local
    minimum is returned. ``maxneighbor`` defaults to
    ``max(250, 1.25 percent of k(n - k))``. Randomness is Philox.

    References
    ----------
    Ng, R. T. and Han, J. (2002). CLARANS: a method for clustering objects
    for spatial data mining. *IEEE Transactions on Knowledge and Data
    Engineering*, 14(5), 1003-1016.

    Examples
    --------
    >>> r = clarans([[0, 0], [0, 1], [1, 0], [8, 8], [8, 9], [9, 8]], 2)
    >>> sorted(r.medoids), r.cluster
    ([1, 4], [1, 1, 1, 2, 2, 2])
    """
    P = _pts(X)
    n = len(P)
    if not 1 <= k < n:
        raise ValueError("need 1 <= k < n")
    D = [[_d(a, b) for b in P] for a in P]
    if maxneighbor is None:
        maxneighbor = max(250, int(0.0125 * k * (n - k)))
    U = _Uniforms(seed)

    def cost(med):
        return ssum(min(D[i][m] for m in med) for i in range(n))

    best, best_cost = None, math.inf
    for _ in range(numlocal):
        cur = []
        while len(cur) < k:
            j = min(int(U.next() * n), n - 1)
            if j not in cur:
                cur.append(j)
        cc = cost(cur)
        fails = 0
        while fails < maxneighbor:
            pos = min(int(U.next() * k), k - 1)
            others = [j for j in range(n) if j not in cur]
            cand = others[min(int(U.next() * len(others)), len(others) - 1)]
            nb = list(cur)
            nb[pos] = cand
            nc = cost(nb)
            if nc < cc:
                cur, cc, fails = nb, nc, 0
            else:
                fails += 1
        if cc < best_cost:
            best, best_cost = sorted(cur), cc
    lab = [min(range(k), key=lambda j: (D[i][best[j]], j)) for i in range(n)]
    return RichResult(payload={"medoids": [m + 1 for m in best], "cluster": _first_appearance(lab), "cost": best_cost})


def cure_clustering(X, k: int, *, n_rep: int = 5, alpha: float = 0.3) -> RichResult:
    r"""CURE hierarchical clustering (Guha, Rastogi and Shim 1998) with shrunken representative points.

    Each cluster keeps up to ``n_rep`` well-scattered points (farthest-point
    selection starting from the point farthest from the mean) shrunk toward
    the mean by ``alpha``; the two clusters whose representatives are
    closest merge until ``k`` remain. ``alpha = 0`` with many representatives
    approaches single linkage, ``alpha = 1`` centroid linkage.

    References
    ----------
    Guha, S., Rastogi, R. and Shim, K. (1998). CURE: an efficient clustering
    algorithm for large databases. *Proc. ACM SIGMOD*, 73-84.

    Examples
    --------
    >>> cure_clustering([[0, 0], [0, 1], [1, 0], [8, 8], [8, 9], [9, 8]], 2).cluster
    [1, 1, 1, 2, 2, 2]
    """
    P = _pts(X)
    n, p = len(P), len(P[0])
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n")

    def reps(mem):
        mu = [ssum(P[i][t] for i in mem) / len(mem) for t in range(p)]
        chosen = []
        for _ in range(min(n_rep, len(mem))):
            if not chosen:
                far = max(mem, key=lambda i: (_d(P[i], mu), -i))
            else:
                far = max((i for i in mem if i not in chosen), key=lambda i: (min(_d(P[i], P[c]) for c in chosen), -i))
            chosen.append(far)
        return [[P[c][t] + alpha * (mu[t] - P[c][t]) for t in range(p)] for c in chosen]

    clusters = [[i] for i in range(n)]
    R = [reps(c) for c in clusters]
    while len(clusters) > k:
        best, bd = None, math.inf
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                dd = min(_d(u, v) for u in R[a] for v in R[b])
                if dd < bd:
                    best, bd = (a, b), dd
        a, b = best
        clusters[a] = sorted(clusters[a] + clusters[b])
        R[a] = reps(clusters[a])
        del clusters[b], R[b]
    lab = [0] * n
    for j, c in enumerate(clusters):
        for i in c:
            lab[i] = j
    return RichResult(payload={"cluster": _first_appearance(lab), "representatives": R})


def diana(D, k: int | None = None, *, is_distance: bool = True) -> RichResult:
    r"""DIANA divisive hierarchical clustering (Kaufman and Rousseeuw 1990), as ``cluster::diana``.

    The cluster of largest diameter is split: the object with the largest
    average dissimilarity to the others starts a splinter group, and objects
    whose average dissimilarity to the rest exceeds that to the splinter
    group (largest difference first) join it. Returns the splits (members
    and diameters, in order of decreasing diameter), the divisive
    coefficient ``dc = mean(1 - d(i) / diam)`` with ``d(i)`` the diameter of
    the last cluster holding object ``i`` before it became a singleton, and,
    with ``k``, the labels after the ``k - 1`` largest splits.

    References
    ----------
    Kaufman, L. and Rousseeuw, P. J. (1990). *Finding Groups in Data*. Wiley, ch. 6.

    Examples
    --------
    >>> r = diana([[0, 1, 5, 6], [1, 0, 4, 5], [5, 4, 0, 1], [6, 5, 1, 0]], 2)
    >>> r.cluster, round(r.dc, 6)
    ([1, 1, 2, 2], 0.833333)
    """
    M = [[float(v) for v in r] for r in D] if is_distance else [[_d(a, b) for b in _pts(D)] for a in _pts(D)]
    n = len(M)

    def diam(c):
        return max((M[i][j] for i in c for j in c), default=0.0)

    splits = []
    last = [0.0] * n
    stack = [list(range(n))]
    while stack:
        c = stack.pop()
        if len(c) < 2:
            continue
        dm = diam(c)
        for i in c:
            last[i] = dm
        avg = [ssum(M[i][j] for j in c if j != i) / (len(c) - 1) for i in c]
        s0 = c[max(range(len(c)), key=lambda t: (avg[t], -t))]
        spl, rest = [s0], [i for i in c if i != s0]
        while len(rest) > 1:
            diff = [
                ssum(M[i][j] for j in rest if j != i) / (len(rest) - 1) - ssum(M[i][j] for j in spl) / len(spl)
                for i in rest
            ]
            t = max(range(len(rest)), key=lambda q: (diff[q], -q))
            if diff[t] <= 0:
                break
            spl.append(rest.pop(t))
        splits.append({"members": sorted(c), "diameter": dm, "parts": (sorted(rest), sorted(spl))})
        stack.extend([sorted(spl), sorted(rest)])
    full = diam(list(range(n)))
    dc = ssum(1 - last[i] / full for i in range(n)) / n if full > 0 else 0.0
    order = sorted(range(len(splits)), key=lambda s: (-splits[s]["diameter"], s))
    splits = [splits[s] for s in order]
    out = {"splits": splits, "dc": dc}
    if k is not None:
        if not 1 <= k <= n:
            raise ValueError("k must be in 1..n")
        lab = [0] * n
        nxt = 1
        for s in splits[: k - 1]:
            for i in s["parts"][1]:
                lab[i] = nxt
            nxt += 1
        out["cluster"] = _first_appearance(lab)
    return RichResult(payload=out)


def _fiedler_split(nodes, W):
    """Split off the first connected component, or bisect at the median of the Fiedler vector of the weighted Laplacian."""
    m = len(nodes)
    inset = set(nodes)
    seen, comp, queue = {nodes[0]}, [nodes[0]], [nodes[0]]
    while queue:
        u = queue.pop(0)
        for v in nodes:
            if v not in seen and v in inset and W[u][v] > 0:
                seen.add(v)
                comp.append(v)
                queue.append(v)
    if len(comp) < m:  # disconnected: a zero edge cut separates the first component
        return [v for v in nodes if v in seen], [v for v in nodes if v not in seen]
    L = [[0.0] * m for _ in range(m)]
    for a in range(m):
        for b in range(m):
            if a != b:
                L[a][b] = -W[nodes[a]][nodes[b]]
        L[a][a] = ssum(W[nodes[a]][nodes[b]] for b in range(m) if b != a)
    vals, vecs = jacobi(L)
    f = [vecs[a][1] for a in range(m)]
    order = sorted(range(m), key=lambda a: (f[a], a))
    half = set(order[: m // 2])
    A = [nodes[a] for a in range(m) if a in half]
    B = [nodes[a] for a in range(m) if a not in half]
    return A, B


def _cut(A, B, W):
    w = [W[i][j] for i in A for j in B if W[i][j] > 0]
    return len(w), ssum(w)


def chameleon(X, k: int, *, n_neighbors: int = 5, min_size: int = 6, alpha: float = 2.0) -> RichResult:
    r"""CHAMELEON (Karypis, Han and Kumar 1999): graph partitioning, then merging by interconnectivity and closeness.

    Phase 1 builds the symmetric ``n_neighbors``-nearest-neighbour graph with
    similarity weights ``1 / (1 + d)`` and recursively bisects it (connected components first, then spectral
    bisection at the median of the Fiedler vector, standing in for hMETIS)
    until every part has at most ``min_size`` points. Phase 2 repeatedly
    merges the connected pair maximising ``RI * RC^alpha`` with relative
    interconnectivity ``RI = |EC(Ci, Cj)| / ((|EC_Ci| + |EC_Cj|) / 2)`` and
    relative closeness ``RC = S(Ci, Cj) / (ni/(ni+nj) S_Ci + nj/(ni+nj) S_Cj)``,
    where ``EC_C`` is the bisection edge cut of ``C``, ``|.|`` its total
    weight and ``S`` a mean cut-edge weight, until ``k`` clusters remain.

    References
    ----------
    Karypis, G., Han, E.-H. and Kumar, V. (1999). CHAMELEON: a hierarchical
    clustering algorithm using dynamic modeling. *IEEE Computer*, 32(8), 68-75.

    Examples
    --------
    >>> pts = [[i * 0.3, (i % 3) * 0.3] for i in range(9)] + [[20 + i * 0.3, (i % 3) * 0.3] for i in range(9)]
    >>> chameleon(pts, 2, n_neighbors=3, min_size=4).cluster == [1] * 9 + [2] * 9
    True
    """
    P = _pts(X)
    n = len(P)
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n")
    D = [[_d(a, b) for b in P] for a in P]
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        nn = sorted((j for j in range(n) if j != i), key=lambda j: (D[i][j], j))[:n_neighbors]
        for j in nn:
            W[i][j] = W[j][i] = 1.0 / (1.0 + D[i][j])
    parts, stack = [], [list(range(n))]
    while stack:
        c = stack.pop()
        if len(c) <= min_size:
            parts.append(sorted(c))
            continue
        A, B = _fiedler_split(c, W)
        stack.extend([B, A])

    def internal(c):
        if len(c) < 2:
            return 0.0, 0.0
        A, B = _fiedler_split(c, W)
        cnt, tot = _cut(A, B, W)
        return tot, (tot / cnt if cnt else 0.0)

    info = [internal(c) for c in parts]
    eps = 1e-12
    while len(parts) > k:
        best, bs = None, -math.inf
        for a in range(len(parts)):
            for b in range(a + 1, len(parts)):
                cnt, tot = _cut(parts[a], parts[b], W)
                if cnt == 0:
                    continue
                na, nb = len(parts[a]), len(parts[b])
                ri = tot / ((info[a][0] + info[b][0]) / 2 + eps)
                rc = (tot / cnt) / (na / (na + nb) * info[a][1] + nb / (na + nb) * info[b][1] + eps)
                sc = ri * rc**alpha
                if sc > bs:
                    best, bs = (a, b), sc
        if best is None:
            break
        a, b = best
        parts[a] = sorted(parts[a] + parts[b])
        info[a] = internal(parts[a])
        del parts[b], info[b]
    lab = [0] * n
    for j, c in enumerate(parts):
        for i in c:
            lab[i] = j
    return RichResult(payload={"cluster": _first_appearance(lab), "n_clusters": len(parts)})


def cheatsheet() -> str:
    return "affinity_propagation / clarans / cure_clustering / diana / chameleon -> clustering of point data."
