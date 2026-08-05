# morie.fn -- function file (rootcoder007/morie)
"""Maximum flow / minimum cut."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["max_flow_min_cut"]


def max_flow_min_cut(G, source, sink):
    """
    Maximum flow / minimum cut

    Formula: Ford-Fulkerson augmentation, with Edmonds-Karp's rule that
    the augmenting path is always a SHORTEST one in the residual graph
    (breadth-first search), which bounds the number of augmentations by
    O(V E) and makes the procedure terminate on irrational capacities as
    well.

    On termination the vertices reachable from the source in the residual
    graph form the source side S of a minimum cut, and the max-flow
    min-cut theorem says cap(S, V\\S) equals the value of the flow.  Both
    numbers are computed here by separate routes and returned, so their
    equality is checkable rather than assumed.

    Parameters
    ----------
    G : array-like
        Square non-negative capacity matrix; G[i][j] is the capacity of
        the directed arc i -> j.
    source, sink : int
        Zero-based vertex indices; must differ.

    Returns
    -------
    result : dict
        Keys: estimate (max flow value), max_flow, min_cut, cut_size,
        source_side, augmentations, n, method.

    References
    ----------
    Ford & Fulkerson (1956), Canadian Journal of Mathematics 8:399-404,
    doi:10.4153/CJM-1956-045-5.
    Edmonds & Karp (1972), JACM 19(2):248-264, doi:10.1145/321694.321699.
    """
    C = core.mat(G)
    n = len(C)
    if n == 0:
        raise ValueError("empty input: G has no vertices")
    if any(len(r) != n for r in C):
        raise ValueError("G must be square")
    for r in C:
        for w in r:
            if w < 0.0:
                raise ValueError("capacities must be non-negative")
    s = int(source)
    t = int(sink)
    if not (0 <= s < n) or not (0 <= t < n):
        raise ValueError("source and sink must be valid vertex indices")
    if s == t:
        raise ValueError("source and sink must differ")
    R = [[C[i][j] for j in range(n)] for i in range(n)]
    flow = 0.0
    aug = 0
    while True:
        prev = [-1] * n
        prev[s] = s
        q = [s]
        while q and prev[t] < 0:
            v = q.pop(0)
            for w in range(n):
                if prev[w] < 0 and R[v][w] > 0.0:
                    prev[w] = v
                    q.append(w)
        if prev[t] < 0:
            break
        # bottleneck along the shortest augmenting path
        b = float("inf")
        w = t
        while w != s:
            v = prev[w]
            if R[v][w] < b:
                b = R[v][w]
            w = v
        w = t
        while w != s:
            v = prev[w]
            R[v][w] -= b
            R[w][v] += b
            w = v
        flow += b
        aug += 1
    # min cut: everything reachable from s in the final residual graph
    seen = [False] * n
    seen[s] = True
    q = [s]
    while q:
        v = q.pop(0)
        for w in range(n):
            if not seen[w] and R[v][w] > 0.0:
                seen[w] = True
                q.append(w)
    side = [i for i in range(n) if seen[i]]
    cut = 0.0
    for i in range(n):
        if not seen[i]:
            continue
        for j in range(n):
            if not seen[j]:
                cut += C[i][j]
    return RichResult(payload={
        "estimate": flow,
        "max_flow": flow,
        "min_cut": cut,
        "cut_size": len(side),
        "source_side": side,
        "augmentations": aug,
        "n": n,
        "method": "Maximum flow / minimum cut",
    })


def cheatsheet():
    return "flowmm: Maximum flow / minimum cut"


# compact alias per ledger/NAMING.md
maxflowmincut = max_flow_min_cut
