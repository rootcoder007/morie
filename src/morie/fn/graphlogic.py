# morie.fn -- function file (rootcoder007/morie)
"""Graphs, satisfiability and sequence labelling: normalised degree centrality, graph diameter
and eccentricities, Brandes betweenness, a conflict-driven clause-learning SAT solver and a
linear-chain conditional random field (forward-backward, Viterbi, L-BFGS training)."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from .lbfgsb import lbfgsb_minimize

__all__ = [
    "graph_degree_centrality",
    "graph_diameter",
    "graph_betweenness",
    "cdcl_solve",
    "crf_fit",
    "crf_viterbi",
    "crf_marginals",
]


def _adj(A):
    return [[float(v) for v in row] for row in A]


def graph_degree_centrality(A, *, mode: str = "all") -> list:
    r"""Freeman degree centrality ``C_D(v) = deg(v) / (n - 1)``.

    ``A`` is an adjacency matrix (nonzero = edge). For a directed graph use
    ``mode="out"`` (row counts), ``"in"`` (column counts) or ``"all"``
    (their sum); for a symmetric ``A`` ``"all"`` counts each edge once.

    References
    ----------
    Freeman, L. C. (1979). Centrality in social networks: conceptual
    clarification. Social Networks 1, 215-239.

    Examples
    --------
    >>> graph_degree_centrality([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    [1.0, 0.5, 0.5]
    """
    M = _adj(A)
    n = len(M)
    sym = all(M[i][j] == M[j][i] for i in range(n) for j in range(n))
    out = []
    for i in range(n):
        o = sum(1 for j in range(n) if M[i][j] != 0 and j != i)
        c = sum(1 for j in range(n) if M[j][i] != 0 and j != i)
        d = o if mode == "out" else c if mode == "in" else (o if sym else o + c)
        out.append(d / (n - 1))
    return out


def _sssp(M, s, weighted):
    # distances, predecessor lists, path counts and settle order from source s
    n = len(M)
    inf = math.inf
    dist = [inf] * n
    sigma = [0.0] * n
    preds = [[] for _ in range(n)]
    dist[s], sigma[s] = 0.0, 1.0
    order = []
    if not weighted:
        queue = [s]
        k = 0
        while k < len(queue):
            v = queue[k]
            k += 1
            order.append(v)
            for w in range(n):
                if M[v][w] != 0 and w != v:
                    if dist[w] == inf:
                        dist[w] = dist[v] + 1.0
                        queue.append(w)
                    if dist[w] == dist[v] + 1.0:
                        sigma[w] += sigma[v]
                        preds[w].append(v)
        return dist, sigma, preds, order
    done = [False] * n
    while True:
        v, best = -1, inf
        for u in range(n):
            if not done[u] and dist[u] < best:
                v, best = u, dist[u]
        if v < 0:
            break
        done[v] = True
        order.append(v)
        for w in range(n):
            if M[v][w] != 0 and w != v and not done[w]:
                nd = dist[v] + M[v][w]
                if nd < dist[w]:
                    dist[w] = nd
                    sigma[w] = sigma[v]
                    preds[w] = [v]
                elif nd == dist[w]:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
    return dist, sigma, preds, order


def graph_diameter(A, *, weighted: bool = False) -> RichResult:
    r"""Graph diameter ``max_{u,v} d(u, v)`` with eccentricities and a farthest pair.

    Shortest paths by breadth-first search (hop counts) or, with
    ``weighted=True``, Dijkstra on the edge lengths ``A[u][v]``. Unreachable
    pairs are ignored (the diameter of the largest finite distances, as
    ``igraph::diameter(unconnected = TRUE)``).

    References
    ----------
    Newman, M. E. J. (2010). Networks: An Introduction, section 6.10.

    Examples
    --------
    >>> r = graph_diameter([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    >>> r.diameter, r.pair
    (3.0, [0, 3])
    """
    M = _adj(A)
    n = len(M)
    ecc, best, pair = [], -1.0, [0, 0]
    for s in range(n):
        d = _sssp(M, s, weighted)[0]
        fin = [x for x in d if x < math.inf]
        e = max(fin)
        ecc.append(e)
        if e > best:
            best = e
            pair = [s, max(range(n), key=lambda j: (d[j] if d[j] < math.inf else -1.0, -j))]
    return RichResult(payload={"diameter": best, "eccentricity": ecc, "pair": pair})


def graph_betweenness(A, *, weighted: bool = False, normalized: bool = False) -> list:
    r"""Betweenness centrality by Brandes' algorithm.

    ``C_B(v) = sum_{s != v != t} sigma_st(v) / sigma_st`` accumulated by
    dependencies ``delta_s(v) = sum_{w: v in P_s(w)} sigma_sv / sigma_sw (1 + delta_s(w))``
    in reverse settle order; halved for symmetric (undirected) ``A``.
    ``normalized`` divides by ``(n - 1)(n - 2) / 2`` (undirected) or
    ``(n - 1)(n - 2)`` (directed), as igraph.

    References
    ----------
    Brandes, U. (2001). A faster algorithm for betweenness centrality. J.
    Math. Sociology 25, 163-177. Freeman, L. C. (1977). Sociometry 40, 35-41.

    Examples
    --------
    >>> graph_betweenness([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    [0.0, 1.0, 0.0]
    """
    M = _adj(A)
    n = len(M)
    sym = all(M[i][j] == M[j][i] for i in range(n) for j in range(n))
    cb = [0.0] * n
    for s in range(n):
        dist, sigma, preds, order = _sssp(M, s, weighted)
        delta = [0.0] * n
        for w in reversed(order):
            for v in preds[w]:
                delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    if sym:
        cb = [v / 2.0 for v in cb]
    if normalized and n > 2:
        den = (n - 1) * (n - 2) / (2.0 if sym else 1.0)
        cb = [v / den for v in cb]
    return cb


def cdcl_solve(cnf, n_vars: int | None = None, *, max_conflicts: int = 100000) -> RichResult:
    r"""Conflict-driven clause learning (CDCL) SAT solver.

    ``cnf`` lists clauses of nonzero integer literals (``v`` / ``-v`` for
    variable ``v`` true / false). Unit propagation by clause scanning, VSIDS
    decisions (activity bumped on learned-clause variables, decayed by 0.95
    per conflict, ties to the lowest index, value false first), first-UIP
    conflict analysis by resolution with reason clauses, non-chronological
    backjumping to the second-highest level of the learned clause.

    References
    ----------
    Marques-Silva, J. P. and Sakallah, K. A. (1999). GRASP: a search
    algorithm for propositional satisfiability. IEEE Trans. Computers 48,
    506-521. Moskewicz, M. W. et al. (2001). Chaff: engineering an efficient
    SAT solver. DAC 2001, 530-535.

    Examples
    --------
    >>> r = cdcl_solve([[1, 2], [-1, 2], [1, -2], [-1, -2, 3]])
    >>> r.satisfiable, r.model
    (True, [1, 2, 3])
    >>> cdcl_solve([[1], [-1]]).satisfiable
    False
    """
    clauses = [list(c) for c in cnf]
    nv = n_vars if n_vars is not None else max((abs(x) for c in clauses for x in c), default=0)
    val = [0] * (nv + 1)
    level = [0] * (nv + 1)
    reason = [-1] * (nv + 1)
    act = [0.0] * (nv + 1)
    trail = []
    stats = {"decisions": 0, "conflicts": 0, "learned": 0}

    def lit_val(x):
        v = val[abs(x)]
        return 0 if v == 0 else (v if x > 0 else -v)

    def assign(x, lev, why):
        val[abs(x)] = 1 if x > 0 else -1
        level[abs(x)] = lev
        reason[abs(x)] = why
        trail.append(x)

    def propagate(lev):
        changed = True
        while changed:
            changed = False
            for ci, c in enumerate(clauses):
                unassigned, sat = [], False
                for x in c:
                    lv = lit_val(x)
                    if lv == 1:
                        sat = True
                        break
                    if lv == 0:
                        unassigned.append(x)
                if sat:
                    continue
                if not unassigned:
                    return ci
                if len(unassigned) == 1:
                    assign(unassigned[0], lev, ci)
                    changed = True
        return -1

    def analyze(ci, lev):
        learned = list(clauses[ci])
        while True:
            cur = [x for x in learned if level[abs(x)] == lev]
            if len(cur) <= 1:
                break
            for t in reversed(trail):
                if -t in learned and level[abs(t)] == lev:
                    pivot = t
                    break
            r = clauses[reason[abs(pivot)]]
            merged = [x for x in learned if x != -pivot]
            for x in r:
                if x != pivot and x not in merged:
                    merged.append(x)
            learned = merged
        others = [level[abs(x)] for x in learned if level[abs(x)] != lev]
        return learned, max(others) if others else 0

    lev = 0
    if any(len(c) == 0 for c in clauses):
        return RichResult(payload={"satisfiable": False, "model": [], **stats})
    while True:
        ci = propagate(lev)
        if ci >= 0:
            stats["conflicts"] += 1
            if lev == 0 or stats["conflicts"] > max_conflicts:
                return RichResult(payload={"satisfiable": False, "model": [], **stats})
            learned, back = analyze(ci, lev)
            for x in learned:
                act[abs(x)] += 1.0
            act = [a * 0.95 for a in act]
            while trail and level[abs(trail[-1])] > back:
                x = trail.pop()
                val[abs(x)] = 0
                reason[abs(x)] = -1
            clauses.append(learned)
            stats["learned"] += 1
            lev = back
            unit = [x for x in learned if lit_val(x) == 0]
            if len(unit) == 1:
                assign(unit[0], lev, len(clauses) - 1)
            continue
        free = [v for v in range(1, nv + 1) if val[v] == 0]
        if not free:
            model = [v if val[v] == 1 else -v for v in range(1, nv + 1)]
            return RichResult(payload={"satisfiable": True, "model": model, **stats})
        v = max(free, key=lambda u: (act[u], -u))
        lev += 1
        stats["decisions"] += 1
        assign(-v, lev, -1)


# ---- linear-chain CRF ----


def _lse(v):
    m = max(v)
    if m == -math.inf:
        return m
    return m + math.log(ssum(math.exp(a - m) for a in v))


def _unpack(theta, K, d):
    W = [theta[k * d : (k + 1) * d] for k in range(K)]
    b = theta[K * d : K * d + K]
    T = [theta[K * d + K + i * K : K * d + K + (i + 1) * K] for i in range(K)]
    return W, b, T


def _emit(W, b, X):
    return [[ssum(W[k][j] * x[j] for j in range(len(x))) + b[k] for k in range(len(W))] for x in X]


def _fb(E, T):
    n, K = len(E), len(E[0])
    al = [list(E[0])]
    for t in range(1, n):
        al.append([E[t][k] + _lse([al[t - 1][i] + T[i][k] for i in range(K)]) for k in range(K)])
    be = [[0.0] * K for _ in range(n)]
    for t in range(n - 2, -1, -1):
        be[t] = [_lse([T[k][j] + E[t + 1][j] + be[t + 1][j] for j in range(K)]) for k in range(K)]
    return al, be, _lse(al[-1])


def crf_marginals(theta, X, n_labels: int) -> RichResult:
    r"""Forward-backward node and edge marginals of a linear-chain CRF.

    ``theta`` packs the emission weights ``W`` (``K x d``, row-major), label
    biases ``b`` (``K``) and transitions ``A`` (``K x K``, row-major); the
    score of labels ``y`` is ``sum_t (W[y_t] x_t + b[y_t]) + sum_t A[y_{t-1}, y_t]``.

    References
    ----------
    Lafferty, J., McCallum, A. and Pereira, F. (2001). Conditional random
    fields: probabilistic models for segmenting and labeling sequence data.
    ICML 2001, 282-289. Sutton, C. and McCallum, A. (2012). An introduction to
    conditional random fields. Found. Trends Mach. Learn. 4, 267-373.

    Examples
    --------
    >>> r = crf_marginals([0.0] * 8, [[1.0], [2.0]], 2)
    >>> r.node[0], round(r.log_z, 12)
    ([0.5, 0.5], 1.38629436112)
    """
    K, d = n_labels, len(X[0])
    W, b, T = _unpack([float(v) for v in theta], K, d)
    E = _emit(W, b, X)
    al, be, lz = _fb(E, T)
    node = [[math.exp(al[t][k] + be[t][k] - lz) for k in range(K)] for t in range(len(X))]
    edge = [
        [[math.exp(al[t - 1][i] + T[i][j] + E[t][j] + be[t][j] - lz) for j in range(K)] for i in range(K)]
        for t in range(1, len(X))
    ]
    return RichResult(payload={"node": node, "edge": edge, "log_z": lz})


def crf_viterbi(theta, X, n_labels: int) -> RichResult:
    r"""Most probable label sequence of a linear-chain CRF by the Viterbi recursion.

    Ties go to the lowest label. ``theta`` as in :func:`crf_marginals`.

    Examples
    --------
    >>> crf_viterbi([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [[1.0], [-2.0]], 2).labels
    [0, 1]
    """
    K, d = n_labels, len(X[0])
    W, b, T = _unpack([float(v) for v in theta], K, d)
    E = _emit(W, b, X)
    n = len(X)
    dp = [list(E[0])]
    bp = [[0] * K]
    for t in range(1, n):
        row, ptr = [], []
        for k in range(K):
            cand = [dp[t - 1][i] + T[i][k] for i in range(K)]
            i = max(range(K), key=lambda j: (cand[j], -j))
            row.append(cand[i] + E[t][k])
            ptr.append(i)
        dp.append(row)
        bp.append(ptr)
    last = max(range(K), key=lambda j: (dp[-1][j], -j))
    path = [last]
    for t in range(n - 1, 0, -1):
        path.append(bp[t][path[-1]])
    return RichResult(payload={"labels": path[::-1], "score": dp[-1][last]})


def crf_fit(sequences, labels, n_labels: int, *, l2: float = 1.0, max_iter: int = 500) -> RichResult:
    r"""Fit a linear-chain CRF by penalised maximum likelihood (L-BFGS).

    Minimises ``-sum_s log p(y_s | X_s) + (l2 / 2) ||theta||^2``; the
    gradient is the expected minus the observed feature counts from the
    forward-backward marginals. ``sequences`` is a list of ``n_s x d``
    feature matrices, ``labels`` the label sequences (``0..K-1``).

    References
    ----------
    Lafferty, McCallum and Pereira (2001). Sha, F. and Pereira, F. (2003).
    Shallow parsing with conditional random fields. HLT-NAACL 2003, 134-141.

    Examples
    --------
    >>> fit = crf_fit([[[1.0], [-1.0], [1.0]]], [[0, 1, 0]], 2, l2=0.1)
    >>> crf_viterbi(fit.theta, [[1.0], [-1.0]], 2).labels
    [0, 1]
    """
    K, d = n_labels, len(sequences[0][0])
    P = K * d + K + K * K

    def fg(theta):
        W, b, T = _unpack(theta, K, d)
        f = 0.5 * l2 * ssum(v * v for v in theta)
        g = [l2 * v for v in theta]
        for X, y in zip(sequences, labels):
            E = _emit(W, b, X)
            al, be, lz = _fb(E, T)
            n = len(X)
            f += lz - (ssum(E[t][y[t]] for t in range(n)) + ssum(T[y[t - 1]][y[t]] for t in range(1, n)))
            for t in range(n):
                for k in range(K):
                    pk = math.exp(al[t][k] + be[t][k] - lz) - (1.0 if y[t] == k else 0.0)
                    for j in range(d):
                        g[k * d + j] += pk * X[t][j]
                    g[K * d + k] += pk
            for t in range(1, n):
                for i in range(K):
                    for j in range(K):
                        pe = math.exp(al[t - 1][i] + T[i][j] + E[t][j] + be[t][j] - lz)
                        g[K * d + K + i * K + j] += pe - (1.0 if (y[t - 1] == i and y[t] == j) else 0.0)
        return f, g

    cache = {}

    def f(theta):
        key = tuple(theta)
        if key not in cache:
            cache.clear()
            cache[key] = fg(list(theta))
        return cache[key][0]

    def grad(theta):
        key = tuple(theta)
        if key not in cache:
            cache.clear()
            cache[key] = fg(list(theta))
        return cache[key][1]

    res = lbfgsb_minimize(f, [0.0] * P, grad=grad, pgtol=1e-10, factr=10.0, max_iter=max_iter)
    theta = [float(v) for v in res.x]
    return RichResult(payload={"theta": theta, "objective": float(res.fun), "n_labels": K, "n_features": d})


def cheatsheet() -> str:
    return (
        "graph_degree_centrality / graph_diameter / graph_betweenness / cdcl_solve / crf_fit / crf_viterbi / "
        "crf_marginals -> graphs, CDCL satisfiability and linear-chain CRFs."
    )

# alias kept from the retired placeholder of the same name
crf_sequence = crf_fit

# alias kept from the retired placeholder of the same name
degree_centrality = graph_degree_centrality

# alias kept from the retired placeholder of the same name
network_between = graph_betweenness
