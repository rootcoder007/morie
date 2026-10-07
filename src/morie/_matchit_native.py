"""Native matching engines shared by :mod:`morie.matching` (the Python arm of rmorie's).

Line-for-line ports of rmorie's R/C++ engines, so the two arms give the same matches:

* :func:`nn_match` -- MatchIt's nearest-neighbour matcher on a scalar distance (the algorithm of
  ``nn_matchC_vec`` / ``find_control_vec`` in MatchIt 4.7, Greifer et al.): without replacement it
  matches in rounds, treated units by decreasing distance; with replacement each treated unit takes
  its ``ratio`` nearest controls in data order. Ties, search order and caliper as in MatchIt.
* :func:`sap_rect` -- rectangular minimum-cost assignment (shortest augmenting paths with duals).
* :func:`full_match` -- optimal full matching as a minimum-weight edge cover.
* :func:`subclass_scoot`, :func:`subclass_weights`, :func:`mm_weights`, :func:`mm_subclass` --
  MatchIt's subclass repair, weighting and subclass numbering.
"""

from __future__ import annotations

import math

__all__ = [
    "full_match",
    "mm_subclass",
    "mm_weights",
    "nn_match",
    "sap_rect",
    "subclass_scoot",
    "subclass_weights",
]


def _stable_order(values, reverse=False):
    """Indices sorting ``values``, ties in index order (R's order(), also with decreasing = TRUE)."""
    idx = list(range(len(values)))
    if reverse:
        idx.sort(key=lambda i: -values[i])
    else:
        idx.sort(key=lambda i: values[i])
    return idx


def _update_bounds(first_c, last_c, ind_d_ord, eligible, treat):
    if not eligible[ind_d_ord[first_c]]:
        for c in range(first_c + 1, last_c + 1):
            if eligible[ind_d_ord[c]] and treat[ind_d_ord[c]] == 0:
                first_c = c
                break
    if not eligible[ind_d_ord[last_c]]:
        for c in range(last_c - 1, first_c - 1, -1):
            if eligible[ind_d_ord[c]] and treat[ind_d_ord[c]] == 0:
                last_c = c
                break
    return first_c, last_c


def _find_controls(t_id, ind_d_ord, match_d_ord, treat, dist, eligible, r, prev, caliper, first_c, last_c, ratio):
    ii = match_d_ord[t_id]
    iil = iir = ii
    min_dist = 0.0
    if r > 1 and prev:
        for m in prev:
            iil = min(iil, match_d_ord[m])
            iir = max(iir, match_d_ord[m])
        if iil == ii:
            min_dist = abs(dist[t_id] - dist[ind_d_ord[iir]])
        elif iir == ii:
            min_dist = abs(dist[t_id] - dist[ind_d_ord[iil]])
        else:
            min_dist = max(abs(dist[t_id] - dist[ind_d_ord[iil]]), abs(dist[t_id] - dist[ind_d_ord[iir]]))
    di = dist[t_id]
    l_stop = r_stop = left = False
    pid: list[int] = []
    pdist: list[float] = []
    nl = nr = 0
    while not l_stop or not r_stop:
        if l_stop:
            left = False
        elif r_stop:
            left = True
        else:
            left = not left
        if left:
            if iil <= first_c or nl == ratio:
                l_stop = True
                continue
            iil -= 1
            iz = ind_d_ord[iil]
        else:
            if iir >= last_c or nr == ratio:
                r_stop = True
                continue
            iir += 1
            iz = ind_d_ord[iir]
        if not eligible[iz] or treat[iz] != 0:
            continue
        if r > 1 and iz in prev:
            continue
        dc = abs(di - dist[iz])
        if dc > caliper:
            if left:
                l_stop = True
            else:
                r_stop = True
            continue
        if dc < min_dist:
            continue
        if len(pid) >= ratio:
            closer = 0
            for d in pdist:
                if d < dc:
                    closer += 1
                    if closer == ratio:
                        break
            if closer >= ratio:
                if left:
                    l_stop = True
                else:
                    r_stop = True
                continue
        pid.append(iz)
        pdist.append(dc)
        if left:
            nl += 1
            if nl == ratio:
                l_stop = True
        else:
            nr += 1
            if nr == ratio:
                r_stop = True
    n = len(pid)
    if n <= 1:
        return pid
    if n <= ratio and all(pdist[k] <= pdist[k + 1] for k in range(n - 1)):
        return pid
    order = sorted(range(n), key=lambda k: pdist[k])
    if n > ratio:
        # std::partial_sort keeps equal distances in heap order; distinct distances (the generic
        # case) give the same top `ratio` either way
        order = order[:ratio]
    return [pid[k] for k in order]


def nn_match(treat, dist, ratio, replace=False, caliper=None):
    """MatchIt's nearest-neighbour match matrix.

    ``treat`` (0/1) and ``dist`` per unit in data order; ``ratio`` per treated unit (data order).
    Returns one list per treated unit of matched control positions (0-based), in match order.
    """
    n = len(treat)
    if len(dist) != n:
        raise ValueError("treat and distance must have the same length")
    for i in range(n):
        if treat[i] not in (0, 1):
            raise ValueError("treat must be 0/1")
        if not math.isfinite(dist[i]):
            raise ValueError(f"distance must be finite (unit {i + 1} is not)")
    ind_focal = [i for i in range(n) if treat[i] == 1]
    nf = len(ind_focal)
    if len(ratio) != nf:
        raise ValueError("ratio must have one entry per treated unit")
    if any(int(r) < 1 for r in ratio):
        raise ValueError("ratio must be positive")
    rows: list[list[int]] = [[] for _ in range(nf)]
    if nf == 0 or nf == n:
        return rows
    ind_d_ord = _stable_order(dist)
    match_d_ord = [0] * n
    for k, u in enumerate(ind_d_ord):
        match_d_ord[u] = k
    cal = (
        (max(dist) - min(dist) + 1)
        if caliper is None or (isinstance(caliper, float) and math.isnan(caliper))
        else caliper
    )
    eligible = [True] * n
    first_c, last_c = 0, n - 1
    if not replace:
        ord_ = _stable_order([dist[u] for u in ind_focal], reverse=True)
        times = [0] * n
        allowed = [1] * n
        for i in range(nf):
            allowed[ind_focal[i]] = int(ratio[i])
        n_elig_c = n - nf
        for r in range(1, max(int(x) for x in ratio) + 1):
            for ti in ord_:
                if int(ratio[ti]) < r:
                    continue
                if n_elig_c == 0:
                    break
                t_id = ind_focal[ti]
                if not eligible[t_id]:
                    continue
                first_c, last_c = _update_bounds(first_c, last_c, ind_d_ord, eligible, treat)
                k = _find_controls(
                    t_id, ind_d_ord, match_d_ord, treat, dist, eligible, r, rows[ti], cal, first_c, last_c, 1
                )
                if not k:
                    eligible[t_id] = False
                    continue
                rows[ti].append(k[0])
                for ck in (k[0], t_id):
                    if not eligible[ck]:
                        continue
                    times[ck] += 1
                    if times[ck] >= allowed[ck]:
                        eligible[ck] = False
                        if treat[ck] == 0:
                            n_elig_c -= 1
    else:
        for ti in range(nf):
            rows[ti] = _find_controls(
                ind_focal[ti],
                ind_d_ord,
                match_d_ord,
                treat,
                dist,
                eligible,
                1,
                rows[ti],
                cal,
                first_c,
                last_c,
                int(ratio[ti]),
            )
    return rows


def sap_rect(cost):
    """Minimum-cost assignment of every row to a distinct column (rows <= cols).

    ``cost`` is a list of row lists. Returns the 0-based column of each row. The same shortest
    augmenting path with duals as rmorie's C++ ``morie_sap_rect`` (Jonker-Volgenant form), so ties
    resolve the same way.
    """
    nt = len(cost)
    nc = len(cost[0]) if nt else 0
    if nt == 0 or nc < nt:
        raise ValueError("assignment needs ncol >= nrow >= 1")
    inf = math.inf
    u = [0.0] * (nt + 1)
    v = [0.0] * (nc + 1)
    way = [0] * (nc + 1)
    p = [0] * (nc + 1)
    for i in range(1, nt + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (nc + 1)
        used = [False] * (nc + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            ci = cost[i0 - 1]
            ui0 = u[i0]
            for j in range(1, nc + 1):
                if used[j]:
                    continue
                cur = ci[j - 1] - ui0 - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(nc + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if not j0:
                break
    out = [-1] * nt
    for j in range(1, nc + 1):
        if p[j] > 0:
            out[p[j] - 1] = j - 1
    return out


def full_match(p, treat):
    """Optimal full matching (minimum-weight edge cover); see rmorie's ``.morie_full_match``.

    Returns ``(subclass, edges)``: a 1-based matched-set label per unit (first appearance order)
    and the (treated, control) unit positions of the cover's edges.
    """
    it = [i for i, t in enumerate(treat) if t == 1]
    ic = [i for i, t in enumerate(treat) if t == 0]
    D = [[abs(p[a] - p[b]) for b in ic] for a in it]
    mt = [min(row) for row in D]
    mc = [min(D[r][c] for r in range(len(it))) for c in range(len(ic))]
    # (mu_t + mu_c) - d, negatives to 0 -- the same order of operations as the R arm
    gain = [[(mt[r] + mc[c]) - D[r][c] for c in range(len(ic))] for r in range(len(it))]
    gain = [[g if g > 0 else 0.0 for g in row] for row in gain]
    flip = len(it) > len(ic)
    G = [list(col) for col in zip(*gain)] if flip else gain
    a = sap_rect([[-g for g in row] for row in G])
    edges = []
    for r, c in enumerate(a):
        if G[r][c] > 0:
            edges.append((c, r) if flip else (r, c))
    cov_t = {e[0] for e in edges}
    cov_c = {e[1] for e in edges}
    for r in range(len(it)):
        if r not in cov_t:
            row = D[r]
            edges.append((r, min(range(len(ic)), key=lambda c: (row[c], c))))
    for c in range(len(ic)):
        if c not in cov_c:
            edges.append((min(range(len(it)), key=lambda r: (D[r][c], r)), c))
    seen = set()
    uniq = []
    for e in edges:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    edges = uniq
    while True:
        dt = [0] * len(it)
        dc = [0] * len(ic)
        for t, c in edges:
            dt[t] += 1
            dc[c] += 1
        red = [k for k, (t, c) in enumerate(edges) if dt[t] >= 2 and dc[c] >= 2]
        if not red:
            break
        del edges[red[0]]
    dt = [0] * len(it)
    dc = [0] * len(ic)
    for t, c in edges:
        dt[t] += 1
        dc[c] += 1
    key = {}
    for t, c in edges:
        k = ("t", t) if (dt[t] >= 2 or dc[c] == 1) else ("c", c)
        key[it[t]] = k
        key[ic[c]] = k
    labels: dict = {}
    sub = []
    for i in range(len(treat)):
        k = key[i]
        if k not in labels:
            labels[k] = len(labels) + 1
        sub.append(labels[k])
    return sub, [(it[t], ic[c]) for t, c in edges]


def subclass_weights(subclass, treat):
    """MatchIt's ATT subclass weights, each group's nonzero weights rescaled to sum to their number."""
    n = len(treat)
    w = [0.0] * n
    t1: dict = {}
    t0: dict = {}
    for s, t in zip(subclass, treat):
        if s is None:
            continue
        (t1 if t == 1 else t0)[s] = (t1 if t == 1 else t0).get(s, 0) + 1
    for i, (s, t) in enumerate(zip(subclass, treat)):
        if s is None:
            continue
        w[i] = 1.0 if t == 1 else t1.get(s, 0) / t0[s]
    for g in (0, 1):
        idx = [i for i in range(n) if treat[i] == g and w[i] > 0]
        if idx:
            tot = sum(w[i] for i in idx)
            for i in idx:
                w[i] = w[i] * len(idx) / tot
    return w


def subclass_scoot(sub, treat, x):
    """MatchIt's ``subclass_scoot`` with ``min.n = 1`` (see rmorie's ``.morie_subclass_scoot``)."""
    usub = sorted(set(sub))
    nsub = len(usub)
    groups = []
    for t in treat:
        if t not in groups:
            groups.append(t)
    counts = {(g, s): 0 for g in groups for s in usub}
    for s, t in zip(sub, treat):
        counts[(t, s)] += 1
    if all(v >= 1 for v in counts.values()):
        return list(sub)
    if any(sum(counts[(g, s)] for s in usub) < nsub for g in groups):
        raise ValueError("not enough units to fit 1 treated and control unit in each subclass")
    s0 = [usub.index(s) for s in sub]
    for g in groups:
        ind = [i for i, t in enumerate(treat) if t == g]
        st = [0] * nsub
        for i in ind:
            st[s0[i]] += 1
        while min(st) <= 0:
            s = st.index(0)
            if s == nsub - 1:
                left = True
            elif s == 0:
                left = False
            else:
                score = 0.0
                for o in range(nsub):
                    if st[o] > 1 and o != s:
                        score += (st[o] - 1) / (o - s)
                left = score <= 0
            s2 = max(k for k in range(s) if st[k] > 0) if left else next(k for k in range(s + 1, nsub) if st[k] > 0)
            cand = [i for i in ind if s0[i] == s2]
            target = max(x[i] for i in cand) if left else min(x[i] for i in cand)
            pick = [i for i in cand if x[i] == target][-1]
            s0[pick] = s
            st[s] += 1
            st[s2] -= 1
    return [usub[k] for k in s0]


def mm_weights(rows, n, idx_t, treat):
    """MatchIt's match-matrix weights (``weights_matrixC``) with ``normalize = TRUE``."""
    w = [0.0] * n
    for r, cu in enumerate(rows):
        if not cu:
            continue
        for c in cu:
            w[c] += 1.0 / len(cu)
        w[idx_t[r]] += 1.0
    for g in (0, 1):
        idx = [i for i in range(n) if treat[i] == g and w[i] > 0]
        if idx:
            tot = sum(w[i] for i in idx)
            for i in idx:
                w[i] = w[i] * len(idx) / tot
    return w


def mm_subclass(rows, n, idx_t):
    """MatchIt's subclass of a match matrix: one set per matched treated unit, in row order."""
    sub: list = [None] * n
    k = 0
    for r, cu in enumerate(rows):
        if not cu:
            continue
        k += 1
        sub[idx_t[r]] = k
        for c in cu:
            sub[c] = k
    return sub
