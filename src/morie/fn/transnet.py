# morie.fn -- function file (rootcoder007/morie)
"""Transport networks: shortest and k-shortest paths, travelling salesman tours, Clarke-Wright vehicle routing,
gravity trip distribution, logit mode choice, BPR traffic assignment (all-or-nothing, user equilibrium,
system optimum) and Webster signal timing."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "shortest_path",
    "k_shortest_paths",
    "tsp_tour",
    "clarke_wright_vrp",
    "gravity_distribution",
    "logit_mode_shares",
    "traffic_assignment",
    "webster_signal",
]


def _edges(edges):
    return [(int(a), int(b), float(w)) for a, b, w in edges]


def _dijkstra(n, E, s, banned_nodes=(), banned_edges=()):
    out = [[] for _ in range(n)]
    for k, (a, b, w) in enumerate(E):
        if k not in banned_edges and a not in banned_nodes and b not in banned_nodes:
            out[a].append((b, w))
    dist = [math.inf] * n
    pred = [-1] * n
    done = [False] * n
    dist[s] = 0.0
    for _ in range(n):
        u = -1
        for v in range(n):
            if not done[v] and dist[v] < math.inf and (u < 0 or dist[v] < dist[u]):
                u = v
        if u < 0:
            break
        done[u] = True
        for v, w in out[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
    return dist, pred


def _path(pred, s, t):
    if s == t:
        return [s]
    if pred[t] < 0:
        return []
    p = [t]
    while p[-1] != s:
        p.append(pred[p[-1]])
    return p[::-1]


def shortest_path(n: int, edges, source: int, target: int | None = None) -> RichResult:
    r"""Dijkstra (1959) single-source shortest paths on a directed graph with nonnegative edge costs.

    ``edges`` are ``(from, to, cost)`` with nodes ``0..n-1`` (add both
    directions for an undirected network; use travel times as costs for the
    fastest path). The unvisited node of least tentative distance (lowest
    index on ties) is settled next and its out-edges relaxed in input order.

    References
    ----------
    Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
    *Numerische Mathematik*, 1, 269-271.

    Examples
    --------
    >>> r = shortest_path(4, [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 4.0), (2, 3, 1.0)], 0, 3)
    >>> r.distance, r.path
    ([0.0, 1.0, 3.0, 4.0], [0, 1, 2, 3])
    """
    E = _edges(edges)
    if any(w < 0 for _, _, w in E):
        raise ValueError("edge costs must be nonnegative")
    dist, pred = _dijkstra(n, E, source)
    out = {"distance": dist, "predecessor": pred}
    if target is not None:
        out["path"] = _path(pred, source, target)
    return RichResult(payload=out)


def _cost(E, p):
    best = {}
    for a, b, w in E:
        best[(a, b)] = min(w, best.get((a, b), math.inf))
    return ssum(best[(p[i], p[i + 1])] for i in range(len(p) - 1))


def k_shortest_paths(n: int, edges, source: int, target: int, k: int) -> RichResult:
    r"""Yen's (1971) algorithm for the ``k`` shortest loopless paths.

    For each spur node of the last accepted path, edges leaving it that are
    used by accepted paths sharing the same root are removed, the root's
    other nodes are banned, and the spur path found by :func:`shortest_path`
    is joined to the root; the cheapest candidate (lexicographically
    smallest node sequence on ties) is accepted next.

    References
    ----------
    Yen, J. Y. (1971). Finding the k shortest loopless paths in a network.
    *Management Science*, 17(11), 712-716.

    Examples
    --------
    >>> E = [(0, 1, 1.0), (1, 3, 1.0), (0, 2, 1.5), (2, 3, 1.0), (1, 2, 0.2)]
    >>> r = k_shortest_paths(4, E, 0, 3, 3)
    >>> r.paths, r.costs
    ([[0, 1, 3], [0, 1, 2, 3], [0, 2, 3]], [2.0, 2.2, 2.5])
    """
    E = _edges(edges)
    dist, pred = _dijkstra(n, E, source)
    first = _path(pred, source, target)
    if not first:
        return RichResult(payload={"paths": [], "costs": []})
    A, B = [first], []
    while len(A) < k:
        last = A[-1]
        for i in range(len(last) - 1):
            spur, root = last[i], last[: i + 1]
            ban_e = set()
            for p in A:
                if p[: i + 1] == root and len(p) > i + 1:
                    ban_e.update(e for e, (a, b, _) in enumerate(E) if a == p[i] and b == p[i + 1])
            ban_n = set(root[:-1])
            d2, p2 = _dijkstra(n, E, spur, ban_n, ban_e)
            tail = _path(p2, spur, target)
            if tail:
                cand = root[:-1] + tail
                if cand not in A and all(cand != c for _, c in B):
                    B.append((_cost(E, cand), cand))
        if not B:
            break
        B.sort(key=lambda t: (t[0], t[1]))
        A.append(B.pop(0)[1])
    return RichResult(payload={"paths": A, "costs": [_cost(E, p) for p in A]})


def _tour_len(D, t):
    return ssum(D[t[i]][t[(i + 1) % len(t)]] for i in range(len(t)))


def tsp_tour(D, *, method: str = "auto") -> RichResult:
    r"""Travelling salesman tour starting and ending at node 0.

    ``exact``: Held-Karp dynamic programming over subsets (``O(n^2 2^n)``,
    used by ``auto`` up to 12 nodes); ``heuristic``: nearest-neighbour
    construction from node 0 improved by 2-opt (first improving exchange in
    lexicographic ``(i, j)`` order, repeated until none improves by more than
    ``1e-12``). ``D`` may be asymmetric for ``exact``.

    References
    ----------
    Held, M. and Karp, R. M. (1962). A dynamic programming approach to
    sequencing problems. *Journal of SIAM*, 10(1), 196-210.
    Croes, G. A. (1958). A method for solving traveling-salesman problems.
    *Operations Research*, 6(6), 791-812.

    Examples
    --------
    >>> D = [[0, 2, 9, 10], [1, 0, 6, 4], [15, 7, 0, 8], [6, 3, 12, 0]]
    >>> r = tsp_tour(D)
    >>> r.tour, r.length
    ([0, 2, 3, 1], 21.0)
    """
    M = [[float(v) for v in row] for row in D]
    n = len(M)
    if method == "auto":
        method = "exact" if n <= 12 else "heuristic"
    if n <= 2:
        t = list(range(n))
        return RichResult(payload={"tour": t, "length": _tour_len(M, t) if n else 0.0})
    if method == "exact":
        C = {(1, 0): (0.0, -1)}
        for mask in range(1, 1 << n):
            if not mask & 1:
                continue
            for j in range(n):
                if (mask, j) not in C:
                    continue
                cj = C[(mask, j)][0]
                for m in range(1, n):
                    if mask & (1 << m):
                        continue
                    key = (mask | (1 << m), m)
                    v = cj + M[j][m]
                    if key not in C or v < C[key][0]:
                        C[key] = (v, j)
        full = (1 << n) - 1
        best, last = math.inf, -1
        for j in range(1, n):
            v = C[(full, j)][0] + M[j][0]
            if v < best:
                best, last = v, j
        tour, mask = [], full
        while last > 0:
            tour.append(last)
            prev = C[(mask, last)][1]
            mask ^= 1 << last
            last = prev
        tour = [0] + tour[::-1]
        return RichResult(payload={"tour": tour, "length": _tour_len(M, tour)})
    if method != "heuristic":
        raise ValueError("method must be auto, exact or heuristic")
    tour, left = [0], set(range(1, n))
    while left:
        c = tour[-1]
        nxt = min(left, key=lambda v: (M[c][v], v))
        tour.append(nxt)
        left.remove(nxt)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b, c, d = tour[i - 1], tour[i], tour[j], tour[(j + 1) % n]
                if M[a][c] + M[b][d] < M[a][b] + M[c][d] - 1e-12:
                    tour[i : j + 1] = tour[i : j + 1][::-1]
                    improved = True
    return RichResult(payload={"tour": tour, "length": _tour_len(M, tour)})


def clarke_wright_vrp(D, demand, capacity: float, *, depot: int = 0) -> RichResult:
    r"""Clarke and Wright (1964) parallel savings heuristic for the capacitated vehicle routing problem.

    Savings ``s_ij = d_0i + d_0j - d_ij`` (symmetric ``D``) are processed in
    decreasing order (ties by ``(i, j)``); routes ending in ``i`` and ``j``
    are merged (reversing as needed) when both are route ends, the routes
    differ and the combined demand fits ``capacity``.

    References
    ----------
    Clarke, G. and Wright, J. W. (1964). Scheduling of vehicles from a
    central depot to a number of delivery points. *Operations Research*,
    12(4), 568-581.

    Examples
    --------
    >>> D = [[0, 4, 4, 5, 5], [4, 0, 1, 7, 8], [4, 1, 0, 7, 7], [5, 7, 7, 0, 1], [5, 8, 7, 1, 0]]
    >>> r = clarke_wright_vrp(D, [0, 1, 1, 1, 1], 2)
    >>> r.routes, r.cost
    ([[1, 2], [3, 4]], 20.0)
    """
    M = [[float(v) for v in row] for row in D]
    q = [float(v) for v in demand]
    n = len(M)
    custs = [i for i in range(n) if i != depot]
    if any(q[i] > capacity for i in custs):
        raise ValueError("a single demand exceeds capacity")
    route = {i: [i] for i in custs}
    load = {i: q[i] for i in custs}
    S = sorted(
        ((M[depot][i] + M[depot][j] - M[i][j], i, j) for a, i in enumerate(custs) for j in custs[a + 1 :]),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    for s, i, j in S:
        if s <= 0:
            break
        ri, rj = route[i], route[j]
        if ri is rj or load[ri[0]] + load[rj[0]] > capacity:
            continue
        if i not in (ri[0], ri[-1]) or j not in (rj[0], rj[-1]):
            continue
        a = ri if ri[-1] == i else ri[::-1]
        b = rj if rj[0] == j else rj[::-1]
        merged = a + b
        tot = load[ri[0]] + load[rj[0]]
        for v in merged:
            route[v] = merged
        load[merged[0]] = tot
        load[merged[-1]] = tot
    seen, routes = set(), []
    for i in custs:
        r = route[i]
        if id(r) not in seen:
            seen.add(id(r))
            routes.append(r if r[0] <= r[-1] else r[::-1])
    routes.sort()
    cost = ssum(M[depot][r[0]] + ssum(M[r[t]][r[t + 1]] for t in range(len(r) - 1)) + M[r[-1]][depot] for r in routes)
    return RichResult(payload={"routes": routes, "cost": cost, "loads": [ssum(q[v] for v in r) for r in routes]})


def gravity_distribution(
    productions, attractions, cost, beta: float, *, deterrence: str = "exp", tol: float = 1e-12, max_iter: int = 10000
) -> RichResult:
    r"""Doubly constrained gravity trip distribution balanced by Furness (iterative proportional fitting).

    ``T_ij = A_i O_i B_j D_j f(c_ij)`` with ``f = exp(-beta c)`` or ``c^-beta``;
    balancing factors are iterated (rows, then columns) until row and column
    totals match within ``tol`` (relative). Productions and attractions must
    have equal totals.

    References
    ----------
    Wilson, A. G. (1970). *Entropy in Urban and Regional Modelling*. Pion.
    Ortuzar, J. de D. and Willumsen, L. G. (2011). *Modelling Transport*,
    4th ed., Wiley, chapter 5.

    Examples
    --------
    >>> r = gravity_distribution([100, 50], [60, 90], [[1, 2], [2, 1]], 0.5)
    >>> [[round(v, 4) for v in row] for row in r.trips]
    [[47.51, 52.49], [12.49, 37.51]]
    """
    Op, Dd = [float(v) for v in productions], [float(v) for v in attractions]
    if abs(ssum(Op) - ssum(Dd)) > 1e-9 * ssum(Op):
        raise ValueError("productions and attractions must have equal totals")
    C = [[float(v) for v in row] for row in cost]
    F = [[math.exp(-beta * c) if deterrence == "exp" else c ** (-beta) for c in row] for row in C]
    n, m = len(Op), len(Dd)
    B = [1.0] * m
    it = 0
    while it < max_iter:
        it += 1
        A = [1 / ssum(B[j] * Dd[j] * F[i][j] for j in range(m)) for i in range(n)]
        B = [1 / ssum(A[i] * Op[i] * F[i][j] for i in range(n)) for j in range(m)]
        T = [[A[i] * Op[i] * B[j] * Dd[j] * F[i][j] for j in range(m)] for i in range(n)]
        err = max(abs(ssum(T[i]) - Op[i]) / Op[i] for i in range(n))
        if err < tol:
            break
    return RichResult(payload={"trips": T, "A": A, "B": B, "iterations": it})


def logit_mode_shares(utilities) -> RichResult:
    r"""Multinomial logit choice probabilities ``P_im = exp(V_im)/sum_m exp(V_im)`` and the logsum ``ln sum exp V``.

    ``utilities`` is zones (or individuals) x modes; computed stably by
    subtracting each row maximum.

    References
    ----------
    McFadden, D. (1974). Conditional logit analysis of qualitative choice
    behavior. In P. Zarembka (ed.), *Frontiers in Econometrics*, 105-142.

    Examples
    --------
    >>> r = logit_mode_shares([[0.0, math.log(3.0)]])
    >>> [round(v, 6) for v in r.shares[0]], round(r.logsum[0], 6)
    ([0.25, 0.75], 1.386294)
    """
    shares, logsum = [], []
    for row in utilities:
        V = [float(v) for v in row]
        mx = max(V)
        e = [math.exp(v - mx) for v in V]
        s = ssum(e)
        shares.append([v / s for v in e])
        logsum.append(mx + math.log(s))
    return RichResult(payload={"shares": shares, "logsum": logsum})


def _bpr(t0, cap, x, a, b):
    return t0 * (1 + a * (x / cap) ** b)


def traffic_assignment(
    n: int,
    links,
    demand,
    *,
    method: str = "ue",
    alpha: float = 0.15,
    beta: float = 4.0,
    max_iter: int = 1000,
    gap: float = 1e-10,
) -> RichResult:
    r"""Static traffic assignment with BPR link costs ``t = t0 (1 + alpha (x/c)^beta)``.

    ``links`` are ``(from, to, t0, capacity)``, ``demand`` are ``(origin,
    destination, flow)``. ``aon``: all-or-nothing loading on free-flow
    shortest paths. ``ue``: Wardrop user equilibrium by Frank-Wolfe on the
    Beckmann objective; ``so``: system optimum by Frank-Wolfe on marginal
    costs ``t + x t'``. Each iteration loads an all-or-nothing auxiliary
    flow at current costs and takes the exact line-search step (bisection of
    the directional derivative to machine precision); iteration stops when
    the relative gap ``(sum t x - sum t y)/sum t x`` falls below ``gap``.

    References
    ----------
    Beckmann, M., McGuire, C. B. and Winsten, C. B. (1956). *Studies in the
    Economics of Transportation*. Yale University Press.
    LeBlanc, L. J., Morlok, E. K. and Pierskalla, W. P. (1975). An efficient
    approach to solving the road network equilibrium traffic assignment
    problem. *Transportation Research*, 9(5), 309-318.

    Examples
    --------
    >>> r = traffic_assignment(2, [(0, 1, 10.0, 100.0), (0, 1, 15.0, 100.0)], [(0, 1, 200.0)], alpha=1.0,
    ...                        beta=1.0)
    >>> [round(v, 6) for v in r.flow]
    [140.0, 60.0]
    """
    L = [(int(a), int(b), float(t0), float(c)) for a, b, t0, c in links]
    OD = [(int(o), int(d), float(f)) for o, d, f in demand]

    def cost(x, marginal):
        out = []
        for (_, _, t0, c), v in zip(L, x):
            t = _bpr(t0, c, v, alpha, beta)
            if marginal:
                t += v * t0 * alpha * beta * v ** (beta - 1) / c**beta if v > 0 else 0.0
            out.append(t)
        return out

    def aon(t):
        y = [0.0] * len(L)
        E = [(a, b, w) for (a, b, _, _), w in zip(L, t)]
        for o in sorted({o for o, _, _ in OD}):
            # the link used into each node is its cheapest parallel link (lowest index on ties)
            _, pred_node = _dijkstra(n, E, o)
            for oo, d, f in OD:
                if oo != o or f == 0:
                    continue
                v = d
                while v != o:
                    u = pred_node[v]
                    k = min((k for k, (a, b, _, _) in enumerate(L) if a == u and b == v), key=lambda k: (t[k], k))
                    y[k] += f
                    v = u
        return y

    marginal = method == "so"
    x = aon(cost([0.0] * len(L), False))
    if method == "aon":
        tt = cost(x, False)
        return RichResult(payload={"flow": x, "time": tt, "iterations": 0, "gap": float("nan")})
    if method not in ("ue", "so"):
        raise ValueError("method must be aon, ue or so")
    rg, it = math.inf, 0
    while it < max_iter:
        it += 1
        t = cost(x, marginal)
        y = aon(t)
        tx = ssum(a * b for a, b in zip(t, x))
        rg = (tx - ssum(a * b for a, b in zip(t, y))) / tx
        if rg < gap:
            break
        lo, hi = 0.0, 1.0
        dvec = [b - a for a, b in zip(x, y)]
        for _ in range(200):
            mid = (lo + hi) / 2
            z = [a + mid * d for a, d in zip(x, dvec)]
            g = ssum(c * d for c, d in zip(cost(z, marginal), dvec))
            if g > 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-16:
                break
        lam = (lo + hi) / 2
        x = [a + lam * d for a, d in zip(x, dvec)]
    tt = cost(x, False)
    return RichResult(
        payload={"flow": x, "time": tt, "total_time": ssum(a * b for a, b in zip(tt, x)), "iterations": it, "gap": rg}
    )


def webster_signal(flows, saturation_flows, lost_time: float) -> RichResult:
    r"""Webster (1958) optimum cycle, green splits and average delay for a fixed-time signal.

    Critical flow ratios ``y_i = q_i/s_i`` (one critical movement per phase,
    ``Y = sum y_i < 1``), optimum cycle ``C0 = (1.5 L + 5)/(1 - Y)`` s,
    effective greens ``g_i = (C0 - L) y_i / Y`` and Webster's delay ``d =
    c (1 - lambda)^2 / (2 (1 - lambda x)) + x^2/(2 q (1 - x)) - 0.65 (c/q^2)^{1/3}
    x^{2 + 5 lambda}`` with ``lambda = g/c``, ``x = q/(lambda s)`` (flows in
    veh/s).

    References
    ----------
    Webster, F. V. (1958). *Traffic Signal Settings*. Road Research
    Technical Paper 39, HMSO, London.

    Examples
    --------
    >>> r = webster_signal([0.25, 0.15], [0.5, 0.5], 10.0)
    >>> round(r.cycle, 6), [round(g, 6) for g in r.green]
    (100.0, [56.25, 33.75])
    """
    q, s = _vec(flows), _vec(saturation_flows)
    y = [a / b for a, b in zip(q, s)]
    Y = ssum(y)
    if Y >= 1:
        raise ValueError("sum of critical flow ratios must be below 1")
    C = (1.5 * lost_time + 5) / (1 - Y)
    g = [(C - lost_time) * v / Y for v in y]
    delay = []
    for qi, si, gi in zip(q, s, g):
        lam = gi / C
        x = qi / (lam * si)
        delay.append(
            C * (1 - lam) ** 2 / (2 * (1 - lam * x))
            + x * x / (2 * qi * (1 - x))
            - 0.65 * (C / qi**2) ** (1 / 3) * x ** (2 + 5 * lam)
        )
    return RichResult(payload={"cycle": C, "green": g, "Y": Y, "delay": delay})


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def cheatsheet() -> str:
    return "shortest_path / k_shortest_paths / tsp_tour / clarke_wright_vrp / traffic_assignment -> transport."
