"""REDCAP regionalization (Guo 2008).

Guo, D. (2008). Regionalization with dynamically constrained agglomerative clustering and
partitioning (REDCAP). International Journal of Geographical Information Science 22, 801-823.
"""

import math

from ._regcore import bound_ok, canonical, neighbours, partition_tree, sqdist
from ._richresult import RichResult

__all__ = ["redcap"]

_LINKAGES = ("single", "complete", "average", "ward")


def _agglomerate(X, nb, linkage, order, stop):
    """Contiguity-constrained agglomeration. Returns (tree edges in merge order, labels when ``stop`` clusters remain)."""
    n = len(X)
    D = [[sqdist(X[i], X[j]) for j in range(n)] for i in range(n)]
    if linkage == "average":
        D = [[math.sqrt(v) for v in r] for r in D]
    members = {i: [i] for i in range(n)}
    of = list(range(n))
    tree = []
    labels_at_stop = None
    edges = sorted({(min(i, j), max(i, j)) for i in range(n) for j in nb[i]}, key=lambda e: (D[e[0]][e[1]], e))

    def link(A, B):
        if order == "first":  # first-order: only the contiguity edges between the clusters count
            return min(D[i][j] for i in A for j in B if j in nbset[i])
        if linkage == "single":
            return min(D[i][j] for i in A for j in B)
        if linkage == "complete":
            return max(D[i][j] for i in A for j in B)
        if linkage == "average":
            s = 0.0
            for i in A:
                for j in B:
                    s += D[i][j]
            return s / (len(A) * len(B))
        # Ward: |A||B| / (|A| + |B|) times the squared distance of the centroids
        d = len(X[0])
        ca = [0.0] * d
        cb = [0.0] * d
        for i in A:
            for q in range(d):
                ca[q] += X[i][q]
        for j in B:
            for q in range(d):
                cb[q] += X[j][q]
        ca = [v / len(A) for v in ca]
        cb = [v / len(B) for v in cb]
        return len(A) * len(B) / (len(A) + len(B)) * sqdist(ca, cb)

    nbset = [set(r) for r in nb]
    while len(members) > 1:
        if len(members) == stop:
            labels_at_stop = canonical(of)
        pairs = {}
        for i, j in edges:
            a, b = of[i], of[j]
            if a != b:
                key = (min(a, b), max(a, b))
                if key not in pairs:
                    pairs[key] = None
        if not pairs:
            raise ValueError("the contiguity graph is not connected")
        best = min(pairs, key=lambda ab: (link(members[ab[0]], members[ab[1]]), ab))
        a, b = best
        tree.append(next((i, j) for i, j in edges if {of[i], of[j]} == {a, b}))  # shortest edge between them
        members[a] += members.pop(b)
        for i in members[a]:
            of[i] = a
    if stop == 1:
        labels_at_stop = [0] * n
    return tree, labels_at_stop


def redcap(X, adjacency, k, linkage="complete", order="full", min_size=1, weights=None, min_weight=None):
    r"""REDCAP: contiguity-constrained agglomeration builds a spanning tree, which is then cut into k regions.

    Clusters merge only when contiguous; their distance is the chosen linkage over all
    member pairs (``order="full"``) or over the contiguity edges between them
    (``order="first"``, single linkage only; this is the minimum spanning tree). Each merge
    adds the shortest contiguity edge between the two clusters to the tree. Single,
    complete and Ward linkage use squared Euclidean distances and average linkage
    Euclidean distances (the GeoDa convention). The tree is cut k - 1 times at the edge of
    largest reduction in the sum of squared deviations, subject to ``min_size`` and
    ``min_weight``.

    Parameters
    ----------
    X : list of attribute vectors
    adjacency : 0/1 matrix or neighbour lists
    k : int
    linkage : {"single", "complete", "average", "ward"}
    order : {"full", "first"}
    min_size, weights, min_weight
        Region floors (count and summed weight).

    Returns
    -------
    RichResult
        Keys: labels, tree, ssd (after each cut), n_regions.

    References
    ----------
    Guo, D. (2008). International Journal of Geographical Information Science 22, 801-823.

    Examples
    --------
    >>> X = [[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]]
    >>> redcap(X, [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]], 2)["labels"]
    [0, 0, 0, 1, 1, 1]
    """
    if linkage not in _LINKAGES or order not in ("full", "first") or (order == "first" and linkage != "single"):
        raise ValueError('linkage must be single, complete, average or ward; order "first" needs single linkage')
    X = [[float(v) for v in r] for r in X]
    n = len(X)
    nb = neighbours(adjacency, n)
    tree, _ = _agglomerate(X, nb, linkage, order, 0)
    labels, hist = partition_tree(X, n, tree, int(k), bound_ok(n, weights, min_size, min_weight))
    return RichResult(
        title="REDCAP regionalization",
        summary_lines=[("regions", max(labels) + 1)],
        payload={"labels": labels, "tree": tree, "ssd": hist, "n_regions": max(labels) + 1},
    )


def cheatsheet():
    return "redcap: REDCAP regionalization (Guo 2008)"
