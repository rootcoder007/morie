r"""Multilevel k-way graph partitioning.

Karypis, G., & Kumar, V. (1998) "A Fast and High Quality Multilevel Scheme
for Partitioning Irregular Graphs", *SIAM Journal on Scientific Computing*
20(1), 359-392 -- the METIS paper.

The problem is to split the vertices into ``k`` parts of nearly equal
weight while cutting as little edge weight as possible. Doing that
directly is hard, so the graph is shrunk, cut while small, and the cut is
carried back up and polished at every size on the way:

**Coarsen.** Repeatedly find a matching (a set of edges no two of which
share a vertex) and collapse each matched pair into one multinode whose
weight is the sum of its parts and whose edges are the union, parallel
edges being added together. Two exact identities follow, and both are
anchored:

.. math::

   W(E_{i+1}) = W(E_i) - W(M_i)      \tag{1}

and the edge-cut of a partition read on the coarse graph equals the
edge-cut of the same partition read on the fine one. Three matching rules
are given in the paper and all three are here:

``"hem"`` (heavy edge, the default)
    Match each unmatched vertex to its unmatched neighbour across the
    heaviest edge. By eq. 1 this takes the most weight out of the coarse
    graph, and a coarse graph with less weight has a smaller cut.
``"rm"`` (random)
    Match to a random unmatched neighbour.
``"lem"`` (light edge)
    The opposite of HEM, which the paper includes because it raises the
    average degree of the coarse graph, and some refinement heuristics
    work better on denser graphs.

**Partition the coarsest graph.** ``"gggp"`` grows a region one vertex at
a time, always taking the frontier vertex whose insertion costs least cut
(4 random starts, the best kept); ``"ggp"`` grows it breadth-first
instead (10 random starts). Both are then refined.

**Uncoarsen.** Project the partition down -- every vertex of a multinode
inherits its part -- and refine at each level. ``"bkl"`` (boundary KL, the
default) considers only boundary vertices; ``"kl"`` considers all of them.
A pass moves the highest-gain vertex from the heavier side, marks it, and
updates its neighbours' gains,

.. math::

   g_v = \sum_{u \notin P(v)} w(v, u) - \sum_{u \in P(v), u \ne v} w(v, u),

stopping once ``x`` = 50 consecutive moves have failed to improve the cut
and rolling back to the best state seen.

``k`` parts are obtained by recursive bisection, which is what this paper
describes; the balance target for each bisection is derived from how many
parts each side still has to produce, so an odd ``k`` splits unevenly on
purpose.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = [
    "graph_clustering",
    "metis_partition",
    "edge_cut",
    "coarsen",
    "match_vertices",
    "kernighan_lin",
    "balance_bisection",
]


# ------------------------------------------------------------ graph type

def _as_graph(A, name="A"):
    """Adjacency matrix or edge list to ``{u: {v: w}}``, symmetric."""
    rows = [list(map(float, r)) for r in np.asarray(A, dtype=float)]
    if not rows:
        raise ValueError("grclus: %s is empty" % name)
    n = len(rows)
    if len(rows[0]) != n:
        raise ValueError("grclus: %s must be a square adjacency matrix "
                         "(got %d x %d)" % (name, n, len(rows[0])))
    adj = [{} for _ in range(n)]
    for i in range(n):
        if len(rows[i]) != n:
            raise ValueError("grclus: %s is ragged" % name)
        for j in range(n):
            w = rows[i][j]
            if w != w or w in (float("inf"), float("-inf")):
                raise ValueError("grclus: %s contains a non-finite value"
                                 % name)
            if w < 0:
                raise ValueError("grclus: edge weights must be "
                                 "non-negative (found %g)" % w)
            if i == j or w == 0.0:
                continue
            if abs(w - rows[j][i]) > 1e-9 * max(1.0, abs(w)):
                raise ValueError("grclus: %s must be symmetric; entry "
                                 "(%d, %d) is %g but (%d, %d) is %g"
                                 % (name, i, j, w, j, i, rows[j][i]))
            adj[i][j] = w
    return adj


def total_edge_weight(adj):
    return sum(sum(nbr.values()) for nbr in adj) / 2.0


def edge_cut(A, parts):
    """Total weight of edges whose endpoints are in different parts."""
    adj = A if isinstance(A, list) and A and isinstance(A[0], dict) \
        else _as_graph(A)
    if len(parts) != len(adj):
        raise ValueError("grclus: the partition has %d entries for %d "
                         "vertices" % (len(parts), len(adj)))
    cut = 0.0
    for u in range(len(adj)):
        for v, w in adj[u].items():
            if parts[u] != parts[v]:
                cut += w
    return cut / 2.0


# ------------------------------------------------------------ coarsening

def match_vertices(adj, scheme="hem", seed=17, order=None):
    """One matching, by the paper's RM, HEM or LEM rule.

    Returns a list ``mate`` where ``mate[u]`` is the vertex ``u`` is
    matched to, or ``u`` itself when it stayed unmatched.
    """
    if scheme not in ("hem", "rm", "lem"):
        raise ValueError("grclus: matching must be 'hem', 'rm' or 'lem'")
    n = len(adj)
    mate = list(range(n))
    matched = [False] * n
    if order is None:
        order = _shuffled(n, seed)
    rng = np.random.default_rng(seed + 1)
    for u in order:
        if matched[u]:
            continue
        cands = [v for v in adj[u] if not matched[v]]
        if not cands:
            continue
        if scheme == "rm":
            v = cands[int(rng.integers(0, len(cands)))]
        elif scheme == "hem":
            best = max(adj[u][v] for v in cands)
            tie = [v for v in cands if adj[u][v] == best]
            v = tie[0] if len(tie) == 1 else tie[int(rng.integers(
                0, len(tie)))]
        else:
            best = min(adj[u][v] for v in cands)
            tie = [v for v in cands if adj[u][v] == best]
            v = tie[0] if len(tie) == 1 else tie[int(rng.integers(
                0, len(tie)))]
        mate[u], mate[v] = v, u
        matched[u] = matched[v] = True
    return mate


def _shuffled(n, seed):
    rng = np.random.default_rng(seed)
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def coarsen(adj, vw, mate):
    """Collapse each matched pair into a multinode.

    Vertex weights add; parallel edges add. Returns
    ``(adj2, vw2, mapping)`` with ``mapping[u]`` the coarse vertex that
    fine vertex ``u`` became part of.
    """
    n = len(adj)
    mapping = [-1] * n
    nxt = 0
    for u in range(n):
        if mapping[u] >= 0:
            continue
        v = mate[u]
        mapping[u] = nxt
        if v != u:
            mapping[v] = nxt
        nxt += 1
    adj2 = [{} for _ in range(nxt)]
    vw2 = [0.0] * nxt
    for u in range(n):
        cu = mapping[u]
        vw2[cu] += vw[u]
        for v, w in adj[u].items():
            cv = mapping[v]
            if cu == cv:
                continue
            adj2[cu][cv] = adj2[cu].get(cv, 0.0) + w
    return adj2, vw2, mapping


# ------------------------------------------------------- initial cut

def _gains(adj, parts, side):
    """g_v = external weight - internal weight, for a bisection."""
    g = [0.0] * len(adj)
    for u in range(len(adj)):
        ext = ins = 0.0
        for v, w in adj[u].items():
            if parts[v] == parts[u]:
                ins += w
            else:
                ext += w
        g[u] = ext - ins
    return g


def _grow_partition(adj, vw, target, seed, greedy=True, n_starts=None):
    """GGGP (greedy) or GGP (breadth-first), best of several starts."""
    n = len(adj)
    if n_starts is None:
        n_starts = 4 if greedy else 10
    rng = np.random.default_rng(seed)
    best_parts, best_cut = None, None
    starts = []
    for _ in range(min(n_starts, n)):
        starts.append(int(rng.integers(0, n)))
    for s in starts:
        parts = [1] * n
        parts[s] = 0
        weight = vw[s]
        frontier = dict((v, None) for v in adj[s])
        while weight < target and frontier:
            if greedy:
                # gain of inserting v into the growing region
                best_v, best_g = None, None
                for v in frontier:
                    ins = sum(w for u, w in adj[v].items()
                              if parts[u] == 0)
                    out = sum(w for u, w in adj[v].items()
                              if parts[u] == 1)
                    g = ins - out
                    if best_g is None or g > best_g:
                        best_g, best_v = g, v
                v = best_v
            else:
                v = next(iter(frontier))          # breadth-first order
            del frontier[v]
            parts[v] = 0
            weight += vw[v]
            for u in adj[v]:
                if parts[u] == 1 and u not in frontier:
                    frontier[u] = None
        c = edge_cut(adj, parts)
        if best_cut is None or c < best_cut:
            best_cut, best_parts = c, parts
    if best_parts is None:
        best_parts = [0 if i * 2 < n else 1 for i in range(n)]
    return best_parts


# ------------------------------------------------------- refinement

def kernighan_lin(adj, vw, parts, target, tolerance=0.03, boundary=True,
                  max_passes=10, patience=50):
    r"""KL refinement of a bisection, in the paper's form.

    Each pass repeatedly moves the highest-gain unmarked vertex out of the
    part that is over its target, updating neighbouring gains, and stops
    after ``patience`` = 50 moves that have not improved the cut; the
    moves after the best state are undone. ``boundary=True`` is the BKL
    variant, which only ever considers vertices with an edge across the
    cut.
    """
    n = len(adj)
    parts = list(parts)
    total_w = sum(vw)
    lo = (target - tolerance * total_w)
    hi = (target + tolerance * total_w)
    best_cut = edge_cut(adj, parts)
    for _ in range(int(max_passes)):
        g = _gains(adj, parts, 0)
        locked = [False] * n
        cur_cut = best_cut
        w0 = sum(vw[i] for i in range(n) if parts[i] == 0)
        seen_best, best_state, since = cur_cut, list(parts), 0
        moved_any = False
        while True:
            cands = []
            for v in range(n):
                if locked[v]:
                    continue
                if boundary and not any(parts[u] != parts[v]
                                        for u in adj[v]):
                    continue
                # A move is allowed if it leaves the partition inside
                # the balance window -- or, when the partition is already
                # outside it, if it moves back toward the target. Without
                # that second clause a bisection that starts unbalanced
                # (region growing can overshoot badly at the coarsest
                # level, where one multinode carries many vertices) can
                # never recover: every move is judged by where it lands,
                # and nothing lands inside.
                nw0 = w0 - vw[v] if parts[v] == 0 else w0 + vw[v]
                if not (lo <= nw0 <= hi):
                    if abs(nw0 - target) >= abs(w0 - target):
                        continue
                cands.append(v)
            if not cands:
                break
            v = max(cands, key=lambda t: g[t])
            cur_cut -= g[v]
            w0 = w0 - vw[v] if parts[v] == 0 else w0 + vw[v]
            parts[v] = 1 - parts[v]
            locked[v] = True
            moved_any = True
            for u, w in adj[v].items():
                if parts[u] == parts[v]:
                    g[u] -= 2.0 * w
                else:
                    g[u] += 2.0 * w
            if cur_cut < seen_best - 1e-12:
                seen_best, best_state, since = cur_cut, list(parts), 0
            else:
                since += 1
                if since >= patience:
                    break
        parts = best_state
        if not moved_any or seen_best >= best_cut - 1e-12:
            best_cut = min(best_cut, seen_best)
            break
        best_cut = seen_best
    return parts, best_cut


# ------------------------------------------------------- the bisection

def balance_bisection(adj, vw, parts, target, tolerance=0.03):
    """Force a bisection inside the balance window, cheapest moves first.

    Refinement optimises the cut subject to balance, but a partition
    projected from a coarser level can arrive already outside the window
    -- one multinode may carry several vertices, so the coarse split that
    looked balanced is not. METIS keeps balancing separate from
    refinement for the same reason. Each step moves the vertex whose
    departure costs the least cut, until the window is met or nothing can
    move.
    """
    n = len(adj)
    parts = list(parts)
    total_w = sum(vw)
    lo, hi = target - tolerance * total_w, target + tolerance * total_w
    guard = 0
    while guard < n:
        guard += 1
        w0 = sum(vw[i] for i in range(n) if parts[i] == 0)
        if lo <= w0 <= hi:
            break
        heavy = 0 if w0 > hi else 1
        g = _gains(adj, parts, 0)
        cands = [v for v in range(n) if parts[v] == heavy]
        if len(cands) <= 1:
            break
        v = max(cands, key=lambda t: g[t])
        parts[v] = 1 - parts[v]
    return parts


def _bisect(adj, vw, target, matching="hem", initial="gggp",
            refinement="bkl", tolerance=0.03, coarsest=20, seed=17):
    levels = []
    cur_adj, cur_vw = adj, list(vw)
    guard = 0
    while len(cur_adj) > max(coarsest, 3) and guard < 100:
        guard += 1
        mate = match_vertices(cur_adj, matching, seed + guard)
        nxt_adj, nxt_vw, mapping = coarsen(cur_adj, cur_vw, mate)
        if len(nxt_adj) >= len(cur_adj):
            break                      # nothing collapsed; stop coarsening
        levels.append((cur_adj, cur_vw, mapping))
        cur_adj, cur_vw = nxt_adj, nxt_vw

    scale = sum(cur_vw) / sum(vw) if sum(vw) else 1.0
    parts = _grow_partition(cur_adj, cur_vw, target * scale, seed,
                            greedy=(initial == "gggp"))
    parts, _ = kernighan_lin(cur_adj, cur_vw, parts, target * scale,
                             tolerance, boundary=(refinement == "bkl"))
    for lvl_adj, lvl_vw, mapping in reversed(levels):
        parts = [parts[mapping[u]] for u in range(len(lvl_adj))]
        scale = sum(lvl_vw) / sum(vw) if sum(vw) else 1.0
        parts = balance_bisection(lvl_adj, lvl_vw, parts, target * scale,
                                  tolerance)
        parts, _ = kernighan_lin(lvl_adj, lvl_vw, parts, target * scale,
                                 tolerance,
                                 boundary=(refinement == "bkl"))
    return balance_bisection(adj, vw, parts, target, tolerance)


def metis_partition(A, k=2, weights=None, matching="hem", initial="gggp",
                    refinement="bkl", tolerance=0.03, coarsest=20,
                    seed=17):
    """Multilevel recursive bisection into ``k`` parts."""
    adj = _as_graph(A)
    n = len(adj)
    k = int(k)
    if k < 1:
        raise ValueError("grclus: k must be at least 1")
    if k > n:
        raise ValueError("grclus: k = %d exceeds the %d vertices"
                         % (k, n))
    if matching not in ("hem", "rm", "lem"):
        raise ValueError("grclus: matching must be 'hem', 'rm' or 'lem'")
    if initial not in ("gggp", "ggp"):
        raise ValueError("grclus: initial must be 'gggp' or 'ggp'")
    if refinement not in ("bkl", "kl"):
        raise ValueError("grclus: refinement must be 'bkl' or 'kl'")
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("grclus: tolerance must lie in [0, 1)")
    if weights is None:
        vw = [1.0] * n
    else:
        vw = [float(t) for t in np.atleast_1d(np.asarray(weights,
                                                         dtype=float))]
        if len(vw) != n:
            raise ValueError("grclus: weights has %d entries for %d "
                             "vertices" % (len(vw), n))
        if any(t <= 0 for t in vw):
            raise ValueError("grclus: vertex weights must be positive")

    parts = [0] * n
    # recursive bisection; each call splits a set of vertices into two
    # groups sized by how many parts each side still owes
    def rec(members, n_parts, label, depth):
        if n_parts <= 1 or len(members) <= 1:
            for u in members:
                parts[u] = label
            return
        left_parts = n_parts // 2
        sub_adj, sub_vw, index = _subgraph(adj, vw, members)
        target = sum(sub_vw) * left_parts / float(n_parts)
        cut = _bisect(sub_adj, sub_vw, target, matching, initial,
                      refinement, tolerance, coarsest, seed + depth)
        left = [members[i] for i in range(len(members)) if cut[i] == 0]
        right = [members[i] for i in range(len(members)) if cut[i] == 1]
        if not left or not right:            # degenerate; split by index
            half = max(1, len(members) * left_parts // n_parts)
            left, right = members[:half], members[half:]
        rec(left, left_parts, label, depth + 1)
        rec(right, n_parts - left_parts, label + left_parts, depth + 1)

    rec(list(range(n)), k, 0, 0)

    sizes = [0] * k
    part_w = [0.0] * k
    for u in range(n):
        sizes[parts[u]] += 1
        part_w[parts[u]] += vw[u]
    cut = edge_cut(adj, parts)
    total = total_edge_weight(adj)
    ideal = sum(vw) / float(k)
    return RichResult(payload={
        "estimate": cut,
        "partition": parts,
        "edge_cut": cut,
        "k": k,
        "sizes": sizes,
        "part_weights": part_w,
        "balance": (max(part_w) / ideal) if ideal > 0 else 1.0,
        "total_edge_weight": total,
        "cut_fraction": (cut / total) if total > 0 else 0.0,
        "matching": matching,
        "initial": initial,
        "refinement": refinement,
        "tolerance": tolerance,
        "n": n,
        "method": ("multilevel recursive bisection (Karypis & Kumar "
                   "1998): %s matching, %s initial partition, %s "
                   "refinement" % (matching.upper(), initial.upper(),
                                   refinement.upper())),
        "note": ("edge_cut is the total weight of edges between parts; "
                 "balance is the heaviest part divided by the ideal "
                 "equal share, so 1.0 is perfect"),
    })


def _subgraph(adj, vw, members):
    index = dict((u, i) for i, u in enumerate(members))
    sub = [{} for _ in members]
    for i, u in enumerate(members):
        for v, w in adj[u].items():
            j = index.get(v)
            if j is not None:
                sub[i][j] = w
    return sub, [vw[u] for u in members], index


def graph_clustering(A, k=2, **kw):
    """K-way graph clustering by multilevel recursive bisection."""
    return metis_partition(A, k, **kw)


def cheatsheet():
    return ("grclus: multilevel graph partitioning (Karypis & Kumar "
            "1998, METIS). Coarsen by matching (hem/rm/lem) so that "
            "W(E_{i+1}) = W(E_i) - W(M_i) and the coarse cut equals the "
            "fine cut; partition the coarsest graph by growing a region "
            "(gggp greedy, 4 starts; ggp breadth-first, 10 starts); "
            "uncoarsen, refining at every level with KL moves of gain "
            "g_v = external - internal weight, stopping after 50 "
            "unproductive moves and rolling back. bkl looks only at "
            "boundary vertices. k parts by recursive bisection.")
