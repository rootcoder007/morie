"""HDBSCAN* density clustering (Campello, Moulavi & Sander 2013)."""

import math

from ._richresult import RichResult

__all__ = ["hdbsc", "hdbscan_labels"]


def _core_distances(D, n, mp):
    # Def. 5: core distance = distance to the mpts-nearest neighbour
    # (including the point itself), i.e. the (mp-1)-th order
    # statistic of the sorted row (row[0] == 0 is the self-distance).
    core = []
    for i in range(n):
        ds = sorted(D[i])
        core.append(ds[min(mp - 1, n - 1)])
    return core


def _mst_mreach(D, core, n):
    # Prim MST of the mutual-reachability graph, Def. 7:
    # d_mreach(x, y) = max(d_core(x), d_core(y), d(x, y)).
    in_tree = [False] * n
    key = [math.inf] * n
    parent = [-1] * n
    key[0] = 0.0
    edges = []
    for _ in range(n):
        u, best = -1, math.inf
        for k in range(n):
            if not in_tree[k] and key[k] < best:
                best, u = key[k], k
        in_tree[u] = True
        if parent[u] != -1:
            edges.append((key[u], parent[u], u))
        for v in range(n):
            if not in_tree[v]:
                w = max(core[u], core[v], D[u][v])
                if w < key[v]:
                    key[v] = w
                    parent[v] = u
    return edges


def _single_linkage(edges, n):
    # single linkage on mutual-reachability distances (Prop. 1):
    # merge in ascending weight into a binary tree; each internal
    # node id >= n carries (left, right, split_distance).
    order = sorted(range(len(edges)),
                   key=lambda i: (edges[i][0], edges[i][1], edges[i][2]))
    par = list(range(n))
    node_of = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    children = {}
    node_size = [1] * (2 * n)
    nid = n
    for oi in order:
        w, a, b = edges[oi]
        ra, rb = find(a), find(b)
        na, nb = node_of[ra], node_of[rb]
        children[nid] = (na, nb, w)
        node_size[nid] = node_size[na] + node_size[nb]
        par[ra] = rb
        node_of[rb] = nid
        nid += 1
    return nid - 1, children, node_size


def _points_under(node, children, n):
    out = []
    stack = [node]
    while stack:
        v = stack.pop()
        if v < n:
            out.append(v)
        else:
            l, r, _ = children[v]
            stack.append(l)
            stack.append(r)
    return out


def hdbsc(X, min_pts=5, min_cluster_size=5, selection="eom", verbose=False):
    """
    HDBSCAN* flat clustering by excess-of-mass stability extraction.

    Campello, Moulavi & Sander (2013), the complete method: core
    distance = distance to the mpts-nearest neighbour (Def. 5);
    mutual reachability d_mreach = max(d_core(x), d_core(y), d(x,y))
    (Def. 7); the density hierarchy is single linkage over the
    mutual-reachability distances (Proposition 1).  The hierarchy is
    CONDENSED with a minimum cluster size (their Algorithm 2): at a
    split, a side with fewer than min_cluster_size points is
    spurious -- those points "fall out" as noise at that density
    level lambda = 1/eps and the parent merely shrinks; only when two
    or more non-spurious sides appear is it a true split into new
    clusters.  Working with lambda = 1/eps, each condensed cluster C
    has a birth level lambda_min(C) and, for each object x_j, a
    leave level lambda_max(x_j, C); the cluster STABILITY is their
    Eq. 3,

        S(C) = sum_{x_j in C} ( lambda_max(x_j, C) - lambda_min(C) ),

    the (relative) excess of mass, so more prominent clusters that
    survive longer score higher.  The flat partition maximizes the
    total selected stability with no two clusters nested (Eq. 4),
    solved by the exact bottom-up recursion of Algorithm 3:
    S_hat(C) = max( S(C), sum over children S_hat(C_child) ); C is
    selected when it beats the best selection in its subtrees, and
    on selecting C its descendants are unselected.  Because clusters
    may be chosen at DIFFERENT density levels, this separates
    clusters of differing density that no single DBSCAN* threshold
    can -- the defining advantage of HDBSCAN over DBSCAN.

    Sources
    -------
    Campello, R. J. G. B., Moulavi, D. & Sander, J. (2013).
    Density-based clustering based on hierarchical density
    estimates. *PAKDD 2013*, LNCS 7819, 160-172, Defs. 5-8,
    Algorithms 1-3, Eqs. 1-5 (local copy fetched-wave3/
    Density-Based Clustering Based on Hierarchical Density
    Estimates.pdf).

    Parameters
    ----------
    X : matrix (n x d)
        Data.
    min_pts : int
        mpts (core-distance neighbour count).
    min_cluster_size : int
        Minimum cluster size for the condensed tree (mclSize).
    selection : {"eom", "leaf"}
        Flat-cluster extraction from the condensed tree.  "eom"
        (default, the paper's recommended method) is the Algorithm 3
        excess-of-mass optimum -- fewest, most prominent clusters,
        each possibly at its own density level.  "leaf" selects every
        leaf of the condensed cluster tree instead (also defined by
        Campello 2013 and standard in implementations): more,
        finer-grained, homogeneous-density clusters.  Both extract
        from the identical hierarchy; only the objective differs.
    verbose : bool
        Print stage progress (core/MST/linkage/condense/extract) --
        useful on large inputs where the MST is O(n^2).

    Returns
    -------
    RichResult
        Keys: labels (-1 = noise), core_distances, n_clusters,
        stabilities ({cluster: S}), condensed_tree (rows
        (parent, child, lambda, child_size)), cluster_tree
        (parent -> [children]).
    """
    if selection not in ("eom", "leaf"):
        raise ValueError("selection must be 'eom' or 'leaf'")
    Xv = [[float(v) for v in row] for row in X]
    n = len(Xv)
    mcs = int(min_cluster_size)
    if n < min_pts or n < 2:
        raise ValueError("need at least min_pts points")
    mp = int(min_pts)

    def _say(msg):
        if verbose:
            print("[hdbsc] " + msg, flush=True)

    _say("distances (n=%d)" % n)
    D = [[math.sqrt(sum((a - b) ** 2 for a, b in zip(Xv[i], Xv[j])))
          for j in range(n)] for i in range(n)]
    _say("core distances")
    core = _core_distances(D, n, mp)
    _say("mutual-reachability MST")
    edges = _mst_mreach(D, core, n)
    _say("single linkage")
    root, children, node_size = _single_linkage(edges, n)
    _say("condense")

    # --- condense (Algorithm 2): emit rows (parent_cluster, child,
    # lambda, child_size); child is a point (size 1) that fell out or
    # a newly born sub-cluster.  Root cluster id = n (first free id). ---
    next_cluster = [n + 1]           # root cluster id = n; new ids > n
    node_to_cluster = {root: n}       # tree node acting as a cluster
    birth = {n: 0.0}                  # lambda_min per cluster
    birth_node = {n: root}            # tree node where cluster is born
    rows = []                         # (parent, child, lambda, size)
    cluster_tree = {n: []}            # cluster -> child clusters
    parent_of = {n: None}
    stack = [root]
    while stack:
        node = stack.pop()
        cid = node_to_cluster[node]
        if node < n:
            continue
        l, r, dist = children[node]
        lam = math.inf if dist <= 0 else 1.0 / dist
        sl = node_size[l]
        sr = node_size[r]
        ok_l = sl >= mcs
        ok_r = sr >= mcs
        if ok_l and ok_r:
            for side, sz in ((l, sl), (r, sr)):
                c = next_cluster[0]
                next_cluster[0] += 1
                node_to_cluster[side] = c
                birth[c] = lam
                birth_node[c] = side
                cluster_tree[c] = []
                cluster_tree[cid].append(c)
                parent_of[c] = cid
                rows.append((cid, c, lam, sz))
                stack.append(side)
        elif not ok_l and not ok_r:
            for p in _points_under(node, children, n):
                rows.append((cid, p, lam, 1))
        else:
            big, small = (l, r) if ok_l else (r, l)
            for p in _points_under(small, children, n):
                rows.append((cid, p, lam, 1))
            node_to_cluster[big] = cid
            stack.append(big)

    # --- stability (Eq. 3): for cluster C,
    # S(C) = sum over rows leaving C of size * (lambda - birth(C)). ---
    stability = {c: 0.0 for c in birth}
    for parent, child, lam, sz in rows:
        if math.isinf(lam):
            continue                  # coincident points, zero density span
        stability[parent] += sz * (lam - birth[parent])

    _say("extract (%s)" % selection)

    def subtree(c):
        out = [c]
        for ch in cluster_tree[c]:
            out.extend(subtree(ch))
        return out

    if selection == "leaf":
        # leaf selection (Campello 2013): every leaf of the condensed
        # cluster tree -- finer, homogeneous-density clusters.
        selected = {c: (c != n and not cluster_tree[c]) for c in birth}
    else:
        # Algorithm 3: bottom-up max excess-of-mass selection (Eqs. 4-5).
        order_c = sorted(birth, key=lambda c: -birth[c])   # deepest first
        s_hat = {}
        selected = {c: True for c in birth if c != n}
        for c in order_c:
            if c == n:
                continue
            kids = cluster_tree[c]
            if not kids:
                s_hat[c] = stability[c]
            else:
                sub = sum(s_hat[k] for k in kids)
                if stability[c] < sub:
                    s_hat[c] = sub
                    selected[c] = False
                else:
                    s_hat[c] = stability[c]
                    for k in kids:
                        for d in subtree(k):
                            selected[d] = False

    chosen = sorted(c for c in birth if c != n and selected.get(c))
    label_of = {c: i for i, c in enumerate(chosen)}
    labels = [-1] * n
    # deepest cluster of each point = the max-birth cluster whose
    # birth-node subtree contains it; its label is the nearest
    # selected ancestor (itself included) in the condensed tree.
    contains = {c: set(_points_under(birth_node[c], children, n))
                for c in birth}
    for p in range(n):
        cand = [c for c in birth if p in contains[c]]
        deep = max(cand, key=lambda c: birth[c])
        cur = deep
        while cur is not None and not selected.get(cur, cur == n and False):
            cur = parent_of[cur]
        if cur is not None and cur in label_of:
            labels[p] = label_of[cur]
    condensed = [(p, ch, la, sz) for (p, ch, la, sz) in rows]
    return RichResult(payload={
        "labels": labels,
        "core_distances": core,
        "n_clusters": len(chosen),
        "stabilities": stability,
        "condensed_tree": condensed,
        "cluster_tree": {k: v for k, v in cluster_tree.items()},
        "min_pts": mp,
        "min_cluster_size": mcs,
        "selection": selection,
        "method": "HDBSCAN* %s extraction (Campello 2013)" % (
            "excess-of-mass" if selection == "eom" else "leaf"),
    })


# long descriptive alias (stub-era name)
hdbscan_labels = hdbsc


def cheatsheet():
    return ("hdbsc: mreach MST + single linkage; condense (mclSize); "
            "S(C)=sum(lam_max-lam_min); bottom-up max-stability select")

# public names resolved by fn/_lazy_map.json
hdbscan = hdbsc
