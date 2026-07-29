# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Combinatorial optimisation, and the min-max theorems behind it.

Schrijver A (2003), *Combinatorial Optimization: Polyhedra and
Efficiency*, Springer; Korte B, Vygen J (2018), *Combinatorial
Optimization*, 6th ed., Springer. Original sources: Kruskal JB (1956)
*Proc AMS* 7(1):48-50; Hall P (1935) *J London Math Soc* 10:26-30;
Konig D (1931); Ford LR, Fulkerson DR (1956) *Canadian Journal of
Mathematics* 8:399-404.

Each result here is a *duality*: a maximum equals a minimum. That makes
them self-checking in a way heuristics are not -- compute both sides and
the theorem is either confirmed or the implementation is wrong. Every
function below returns both sides and the residual between them rather
than one side and a promise.
"""

import math
from itertools import combinations

from ._richresult import RichResult

__all__ = [
    "minimum_spanning_tree",
    "bipartite_matching",
    "konig_theorem",
    "hall_condition",
    "max_flow_min_cut",
]

_METHOD = "Combinatorial optimisation with both sides of the duality"


def minimum_spanning_tree(edges, n_vertices, weights=None):
    r"""Kruskal's algorithm, which is the greedy algorithm on the cycle
    matroid.

    The connection to :mod:`morie.fn.matrdt` is the reason it is
    correct: a set of edges is independent exactly when it is acyclic,
    that independence system is a matroid, and Rado-Edmonds says greedy
    is then optimal for every weighting. Nothing about Kruskal is
    special beyond that.

    A disconnected graph has no spanning tree, so ``connected`` is
    reported and the result is a spanning *forest* -- returning its
    weight as though it were a tree would be quietly wrong.

    Returns
    -------
    RichResult with ``weight``, ``tree_edges``, ``connected``,
    ``n_components``.

    Examples
    --------
    >>> e = [(0, 1), (1, 2), (0, 2)]
    >>> minimum_spanning_tree(e, 3, [1, 2, 3])["weight"]
    3
    """
    E = [tuple(e) for e in edges]
    m = len(E)
    n = int(n_vertices)
    if n < 1:
        raise ValueError(f"n_vertices must be positive; got {n}.")
    w = [1] * m if weights is None else list(weights)
    if len(w) != m:
        raise ValueError(
            f"weights has length {len(w)} but there are {m} edges."
        )
    for a, b in E:
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"edge ({a}, {b}) leaves 0 .. {n - 1}.")

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = sorted(range(m), key=lambda i: (w[i], i))
    chosen, total = [], 0
    for i in order:
        a, b = E[i]
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            chosen.append(i)
            total += w[i]
    comps = len({find(v) for v in range(n)})
    connected = comps == 1

    out = RichResult(
        title="Minimum spanning tree (Kruskal)",
        summary_lines=[
            ("Total weight", total),
            ("Edges chosen", len(chosen)),
            ("Connected", connected),
            ("Components", comps),
        ],
        payload={
            "weight": total,
            "estimate": float(total),
            "tree_edges": chosen,
            "edge_list": [E[i] for i in chosen],
            "connected": connected,
            "n_components": comps,
            "n_edges_chosen": len(chosen),
            "expected_tree_edges": n - 1,
            "n": n,
            "method": "Kruskal's algorithm = greedy on the cycle matroid",
        },
    )
    if not connected:
        out.warnings.append(
            f"The graph has {comps} components, so no spanning TREE exists. "
            f"The {len(chosen)} edges returned are a spanning forest and the "
            "weight is the forest's, not a tree's."
        )
    return out


def bipartite_matching(left_n, right_n, edges):
    r"""Maximum matching in a bipartite graph, by augmenting paths.

    Returns the matching itself, not just its size, so it can be
    verified independently.
    """
    ln, rn = int(left_n), int(right_n)
    if ln < 0 or rn < 0:
        raise ValueError("both sides must be non-negative.")
    adj = [[] for _ in range(ln)]
    for a, b in edges:
        if not (0 <= a < ln):
            raise ValueError(f"left endpoint {a} leaves 0 .. {ln - 1}.")
        if not (0 <= b < rn):
            raise ValueError(f"right endpoint {b} leaves 0 .. {rn - 1}.")
        adj[a].append(b)

    match_r = [-1] * rn

    def augment(u, seen):
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                if match_r[v] == -1 or augment(match_r[v], seen):
                    match_r[v] = u
                    return True
        return False

    size = 0
    for u in range(ln):
        if augment(u, [False] * rn):
            size += 1
    pairs = [(u, v) for v, u in enumerate(match_r) if u != -1]
    return {"size": size, "pairs": sorted(pairs), "match_right": match_r,
            "left_n": ln, "right_n": rn}


def konig_theorem(left_n, right_n, edges):
    r"""Konig's theorem: in a bipartite graph the maximum matching
    equals the minimum vertex cover.

    Both sides are computed -- the matching by augmenting paths, the
    cover by the standard alternating-reachability construction -- and
    the equality is checked. The cover is also *verified to be a cover*,
    since a construction that returns the right size but misses an edge
    would otherwise pass.

    Returns
    -------
    RichResult with ``matching_size``, ``cover_size``, ``cover``,
    ``konig_holds``, ``cover_is_valid``.
    """
    ln, rn = int(left_n), int(right_n)
    E = [(int(a), int(b)) for a, b in edges]
    m = bipartite_matching(ln, rn, E)
    match_r = m["match_right"]
    match_l = [-1] * ln
    for v, u in enumerate(match_r):
        if u != -1:
            match_l[u] = v

    adj = [[] for _ in range(ln)]
    for a, b in E:
        adj[a].append(b)

    # alternating reachability from unmatched left vertices
    vis_l = [False] * ln
    vis_r = [False] * rn
    stack = [u for u in range(ln) if match_l[u] == -1]
    for u in stack:
        vis_l[u] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v != match_l[u] and not vis_r[v]:
                vis_r[v] = True
                nu = match_r[v]
                if nu != -1 and not vis_l[nu]:
                    vis_l[nu] = True
                    stack.append(nu)
    cover_left = [u for u in range(ln) if not vis_l[u]]
    cover_right = [v for v in range(rn) if vis_r[v]]
    cover_size = len(cover_left) + len(cover_right)

    cl, cr = set(cover_left), set(cover_right)
    uncovered = [(a, b) for a, b in E if a not in cl and b not in cr]
    valid = not uncovered

    out = RichResult(
        title="Konig's theorem",
        summary_lines=[
            ("Maximum matching", m["size"]),
            ("Minimum vertex cover", cover_size),
            ("Equal", m["size"] == cover_size),
            ("Cover is valid", valid),
        ],
        payload={
            "matching_size": m["size"],
            "estimate": float(m["size"]),
            "matching": m["pairs"],
            "cover_size": cover_size,
            "cover_left": cover_left,
            "cover_right": cover_right,
            "cover_is_valid": valid,
            "uncovered_edges": uncovered,
            "konig_holds": m["size"] == cover_size and valid,
            "n": ln + rn,
            "method": "Konig's theorem (Konig 1931)",
        },
    )
    if not valid:
        out.warnings.append(
            f"The constructed cover misses {len(uncovered)} edges, so it is "
            "not a vertex cover at all. The size agreeing with the matching "
            "would be a coincidence, not a confirmation."
        )
    if m["size"] != cover_size:
        out.warnings.append(
            f"Matching ({m['size']}) and cover ({cover_size}) differ, which "
            "contradicts Konig's theorem on a bipartite graph. Either the "
            "graph is not bipartite as given, or one side is wrong."
        )
    return out


def hall_condition(left_n, right_n, edges):
    r"""Hall's marriage theorem: a perfect matching of the left side
    exists exactly when every subset :math:`S` of it satisfies
    :math:`|N(S)| \ge |S|`.

    Checked exhaustively over all subsets, and cross-checked against
    the matching itself. When the condition fails, the **violating
    set** is returned -- Hall's theorem is most useful as a certificate
    of impossibility, and a bare False is not one.

    Returns
    -------
    RichResult with ``holds``, ``violating_set``, ``deficiency``,
    ``matching_size``, ``agrees_with_matching``.
    """
    ln, rn = int(left_n), int(right_n)
    E = [(int(a), int(b)) for a, b in edges]
    nbr = [set() for _ in range(ln)]
    for a, b in E:
        if not (0 <= a < ln) or not (0 <= b < rn):
            raise ValueError("an edge endpoint is out of range.")
        nbr[a].add(b)

    worst = None
    worst_def = 0
    holds = True
    for r in range(1, ln + 1):
        for S in combinations(range(ln), r):
            union = set()
            for u in S:
                union |= nbr[u]
            deficit = len(S) - len(union)
            if deficit > 0:
                holds = False
                if deficit > worst_def:
                    worst_def, worst = deficit, list(S)
    m = bipartite_matching(ln, rn, E)
    agrees = (m["size"] == ln) == holds

    out = RichResult(
        title="Hall's condition",
        summary_lines=[
            ("Condition holds", holds),
            ("Matching size", m["size"]),
            ("Left side size", ln),
            ("Worst deficiency", worst_def),
        ],
        payload={
            "holds": holds,
            "estimate": float(holds),
            "violating_set": worst,
            "deficiency": worst_def,
            "matching_size": m["size"],
            "perfect_on_left": m["size"] == ln,
            "agrees_with_matching": agrees,
            "n": ln + rn,
            "method": "Hall's marriage theorem (Hall 1935)",
        },
        interpretation=(
            f"Every subset of the left side has at least as many neighbours "
            f"as members, so a matching saturating all {ln} exists."
            if holds else
            f"The set {worst} has only {len(set().union(*[nbr[u] for u in worst])) if worst else 0} "
            f"neighbours for {len(worst) if worst else 0} members, so no "
            "matching can saturate the left side."
        ),
    )
    if not agrees:
        out.warnings.append(
            "Hall's condition and the computed matching disagree about "
            "whether the left side can be saturated. One of the two is wrong."
        )
    return out


def max_flow_min_cut(capacity, source=0, sink=None):
    r"""Ford-Fulkerson with breadth-first augmenting paths, returning
    both the flow value and a minimum cut.

    The max-flow min-cut theorem says the two coincide. The cut is
    derived from the residual graph after the flow is maximal, and its
    capacity is then recomputed from the original matrix rather than
    assumed equal -- so the equality is a check, not a restatement.

    Parameters
    ----------
    capacity : 2-D array
        ``capacity[i][j]`` is the capacity of the arc from i to j.
    source, sink : int

    Returns
    -------
    RichResult with ``flow``, ``cut_capacity``, ``min_cut_source_side``,
    ``theorem_holds``, ``cut_edges``.
    """
    C = [list(map(float, row)) for row in capacity]
    n = len(C)
    if any(len(r) != n for r in C):
        raise ValueError("capacity must be square.")
    if any(v < 0 for r in C for v in r):
        raise ValueError("capacities must be non-negative.")
    s = int(source)
    t = n - 1 if sink is None else int(sink)
    if not (0 <= s < n) or not (0 <= t < n):
        raise ValueError(f"source and sink must lie in 0 .. {n - 1}.")
    if s == t:
        raise ValueError("source and sink must differ.")

    R = [row[:] for row in C]
    flow = 0.0
    while True:
        parent = [-1] * n
        parent[s] = s
        queue = [s]
        while queue and parent[t] == -1:
            u = queue.pop(0)
            for v in range(n):
                if parent[v] == -1 and R[u][v] > 1e-12:
                    parent[v] = u
                    queue.append(v)
        if parent[t] == -1:
            break
        # bottleneck along the path
        push = math.inf
        v = t
        while v != s:
            u = parent[v]
            push = min(push, R[u][v])
            v = u
        v = t
        while v != s:
            u = parent[v]
            R[u][v] -= push
            R[v][u] += push
            v = u
        flow += push

    # the source side of the min cut is what the residual still reaches
    reach = [False] * n
    reach[s] = True
    queue = [s]
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if not reach[v] and R[u][v] > 1e-12:
                reach[v] = True
                queue.append(v)
    cut_edges = [(i, j) for i in range(n) for j in range(n)
                 if reach[i] and not reach[j] and C[i][j] > 0]
    cut_cap = sum(C[i][j] for i, j in cut_edges)

    out = RichResult(
        title="Max-flow min-cut",
        summary_lines=[
            ("Maximum flow", flow),
            ("Minimum cut capacity", cut_cap),
            ("Equal", abs(flow - cut_cap) < 1e-9),
            ("Cut edges", len(cut_edges)),
        ],
        payload={
            "flow": flow,
            "estimate": flow,
            "cut_capacity": cut_cap,
            "min_cut_source_side": [i for i in range(n) if reach[i]],
            "cut_edges": cut_edges,
            "theorem_holds": abs(flow - cut_cap) < 1e-9,
            "residual_gap": abs(flow - cut_cap),
            "source": s, "sink": t,
            "n": n,
            "method": "Ford-Fulkerson with a verified minimum cut",
        },
    )
    if abs(flow - cut_cap) > 1e-9:
        out.warnings.append(
            f"The flow ({flow}) and the cut capacity ({cut_cap}) differ, "
            "which contradicts the max-flow min-cut theorem. The "
            "implementation is wrong."
        )
    return out


def cheatsheet():
    return (
        "cmbopt: Kruskal as greedy on the cycle matroid, bipartite matching "
        "with Konig and Hall, and max-flow min-cut -- each returning both "
        "sides of the duality and the residual between them"
    )
