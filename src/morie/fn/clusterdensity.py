# morie.fn -- function file (rootcoder007/morie)
"""Density and prototype clustering: DENCLUE, FLAME, possibilistic fuzzy c-means and growing neural gas."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform
from .clusops import fuzzy_cmeans

__all__ = ["denclue", "flame_clustering", "possibilistic_fcm", "growing_neural_gas"]


def _pts(X):
    P = [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in X]
    if not P:
        raise ValueError("X must have rows")
    return P


def _d2(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += (x - y) * (x - y)
    return s


def _first_appearance(lab):
    seen, out = {}, []
    for v in lab:
        if v == 0:
            out.append(0)
            continue
        if v not in seen:
            seen[v] = len(seen) + 1
        out.append(seen[v])
    return out


def denclue(X, h: float, xi: float, *, eps: float | None = None, tol: float = 1e-10, maxit: int = 1000) -> RichResult:
    r"""DENCLUE 2.0 (Hinneburg and Gabriel 2007): Gaussian-kernel hill climbing to density attractors.

    The density is ``f(x) = (1/n) sum_i (2 pi h^2)^{-d/2} exp(-||x - x_i||^2 / (2 h^2))``;
    each point climbs by the fixed-point step ``x <- sum_i K(x, x_i) x_i / sum_i K(x, x_i)``
    until the relative density gain is below ``tol``. Points whose attractor
    has density below ``xi`` are noise (label 0); attractors closer than
    ``eps`` (default ``h / 2``) are joined (single linkage) into clusters.

    References
    ----------
    Hinneburg, A. and Keim, D. A. (1998). An efficient approach to clustering
    in large multimedia databases with noise. *Proc. KDD*, 58-65.
    Hinneburg, A. and Gabriel, H.-H. (2007). DENCLUE 2.0: fast clustering
    based on kernel density estimation. *Proc. IDA*, 70-80.

    Examples
    --------
    >>> denclue([[0.0], [0.3], [0.1], [9.0], [9.2], [30.0]], 0.5, 0.2).cluster
    [1, 1, 1, 2, 2, 0]
    """
    P = _pts(X)
    n, d = len(P), len(P[0])
    c = (2 * math.pi * h * h) ** (-d / 2) / n
    eps = h / 2 if eps is None else eps

    def dens(y):
        return c * ssum(math.exp(-_d2(y, q) / (2 * h * h)) for q in P)

    att, fatt = [], []
    for x in P:
        y = list(x)
        fy = dens(y)
        for _ in range(maxit):
            w = [math.exp(-_d2(y, q) / (2 * h * h)) for q in P]
            sw = ssum(w)
            y = [ssum(w[i] * P[i][t] for i in range(n)) / sw for t in range(d)]
            fn = dens(y)
            gain = fn - fy
            fy = fn
            if gain <= tol * fn:
                break
        att.append(y)
        fatt.append(fy)
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    keep = [fatt[i] >= xi for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if keep[i] and keep[j] and _d2(att[i], att[j]) < eps * eps:
                a, b = find(i), find(j)
                par[max(a, b)] = min(a, b)
    lab = [find(i) + 1 if keep[i] else 0 for i in range(n)]
    return RichResult(payload={"cluster": _first_appearance(lab), "attractors": att, "density": fatt})


def flame_clustering(
    X, *, knn: int = 10, outlier_threshold: float = -2.0, steps: int = 500, epsilon: float = 1e-6
) -> RichResult:
    r"""FLAME (Fu and Medico 2007): fuzzy clustering by local approximation of memberships.

    Following the reference C implementation: the ``knn`` nearest neighbours
    (extended over ties of the ``knn``-th distance) get rank weights
    ``(k - j) / (k (k + 1) / 2)``; density is ``1 / sum of neighbour
    distances``. Cluster supporting objects (CSOs) have density at least that
    of every neighbour; outliers have density no higher than any neighbour
    and below ``mean + outlier_threshold * sd`` of the densities. CSOs keep
    full membership of their own cluster, outliers of the outlier group,
    and the rest iterate ``p_i <- sum_j w_ij p_j`` (normalised) until the
    squared change is below ``epsilon``; objects join the group of largest
    membership (outlier group labelled 0).

    References
    ----------
    Fu, L. and Medico, E. (2007). FLAME, a novel fuzzy clustering method for
    the analysis of DNA microarray data. *BMC Bioinformatics*, 8, 3.

    Examples
    --------
    >>> pts = [[0.1 * i + 0.01 * (i % 3), 0.1 * (i * 7 % 5)] for i in range(8)]
    >>> r = flame_clustering(pts + [[x + 10, y] for x, y in pts], knn=5)
    >>> r.cluster == [1] * 8 + [2] * 8
    True
    """
    P = _pts(X)
    n = len(P)
    kmax = min(int(math.sqrt(n)) + 10, n - 1)
    knn = min(knn, kmax)
    graph, dists = [], []
    for i in range(n):
        order = sorted((j for j in range(n) if j != i), key=lambda j: (_d2(P[i], P[j]), j))[:kmax]
        graph.append(order)
        dists.append([math.sqrt(_d2(P[i], P[j])) for j in order])
    counts, weights, density = [], [], []
    for i in range(n):
        k = knn
        dk = dists[i][knn - 1]
        for j in range(knn, kmax):
            if dists[i][j] == dk:
                k += 1
            else:
                break
        counts.append(k)
        s = 0.5 * k * (k + 1.0)
        weights.append([(k - j) / s for j in range(k)])
        density.append(1.0 / (ssum(dists[i][:k]) + 1e-9))
    mu = ssum(density) / n
    thd = mu + outlier_threshold * math.sqrt(max(ssum(v * v for v in density) / n - mu * mu, 0.0))
    typ = [0] * n  # 0 normal, 1 CSO, 2 outlier
    for i in range(n):
        k = counts[i]
        fmax = 0.0
        fmin = density[i] / density[graph[i][0]]
        for j in range(1, k):
            r = density[i] / density[graph[i][j]]
            fmax = max(fmax, r)
            fmin = min(fmin, r)
            if typ[graph[i][j]]:
                fmin = 0.0
        if fmin >= 1.0:
            typ[i] = 1
        elif fmax <= 1.0 and density[i] < thd:
            typ[i] = 2
    csos = [i for i in range(n) if typ[i] == 1]
    m = len(csos)
    A = [[0.0] * (m + 1) for _ in range(n)]
    for i in range(n):
        if typ[i] == 1:
            A[i][csos.index(i)] = 1.0
        elif typ[i] == 2:
            A[i][m] = 1.0
        else:
            A[i] = [1.0 / (m + 1)] * (m + 1)
    B = [list(r) for r in A]
    even = False
    it = 0
    for it in range(1, steps + 1):  # noqa: B007 (reported)
        dev = 0.0
        cur, prev = (B, A) if even else (A, B)
        for i in range(n):
            if typ[i]:
                continue
            k, ids, wt = counts[i], graph[i], weights[i]
            tot = 0.0
            for j in range(m + 1):
                v = 0.0
                for q in range(k):
                    v += wt[q] * prev[ids[q]][j]
                cur[i][j] = v
                dev += (v - prev[i][j]) ** 2
                tot += v
            for j in range(m + 1):
                cur[i][j] /= tot
        even = not even
        if dev < epsilon:
            break
    for i in range(n):  # final smoothing pass of the reference code (A from B)
        k, ids, wt = counts[i], graph[i], weights[i]
        for j in range(m + 1):
            v = 0.0
            for q in range(k):
                v += wt[q] * B[ids[q]][j]
            A[i][j] = v
    lab = []
    for i in range(n):
        j = max(range(m + 1), key=lambda t: (A[i][t], -t))
        lab.append(0 if j == m else j + 1)
    return RichResult(
        payload={
            "cluster": _first_appearance(lab),
            "membership": A,
            "supports": [c + 1 for c in csos],
            "outliers": [i + 1 for i in range(n) if typ[i] == 2],
            "iterations": it,
        }
    )


def possibilistic_fcm(
    X,
    centers,
    *,
    a: float = 1.0,
    b: float = 1.0,
    m: float = 2.0,
    eta: float = 2.0,
    K: float = 1.0,
    tol: float = 1e-9,
    maxit: int = 1000,
) -> RichResult:
    r"""Possibilistic fuzzy c-means (Pal, Pal, Keller and Bezdek 2005).

    Minimises ``sum_ik (a u_ik^m + b t_ik^eta) D_ik^2 + sum_i gamma_i sum_k (1 - t_ik)^eta``
    with memberships ``u`` (rows sum to one) and typicalities ``t``:
    ``u_ik = 1 / sum_j (D_ik / D_jk)^{2/(m-1)}``,
    ``t_ik = 1 / (1 + (b D_ik^2 / gamma_i)^{1/(eta-1)})`` and
    ``v_i = sum_k (a u^m + b t^eta) x_k / sum_k (a u^m + b t^eta)``.
    ``gamma_i = K sum_k u_ik^m D_ik^2 / sum_k u_ik^m`` from a fuzzy c-means
    run started at ``centers`` (Krishnapuram and Keller), which also starts
    the iteration.

    References
    ----------
    Pal, N. R., Pal, K., Keller, J. M. and Bezdek, J. C. (2005). A
    possibilistic fuzzy c-means clustering algorithm. *IEEE Transactions on
    Fuzzy Systems*, 13(4), 517-530.

    Examples
    --------
    >>> r = possibilistic_fcm([[0.0], [1.0], [9.0], [10.0], [4.0]], [[0.0], [10.0]])
    >>> r.cluster
    [1, 1, 2, 2, 1]
    """
    P = _pts(X)
    n, p = len(P), len(P[0])
    f = fuzzy_cmeans(P, centers, m=m)
    V = [list(v) for v in f.centers]
    k = len(V)
    U0 = f.membership
    gam = []
    for c in range(k):
        num = ssum(U0[i][c] ** m * _d2(P[i], V[c]) for i in range(n))
        gam.append(K * num / ssum(U0[i][c] ** m for i in range(n)))
    e = 2.0 / (m - 1.0)
    it = 0
    U, T = U0, []
    for it in range(1, maxit + 1):  # noqa: B007 (reported)
        D2 = [[_d2(P[i], V[c]) for c in range(k)] for i in range(n)]
        U = []
        for i in range(n):
            if any(v == 0 for v in D2[i]):
                z = [1.0 if v == 0 else 0.0 for v in D2[i]]
                s = ssum(z)
                U.append([v / s for v in z])
            else:
                U.append([1.0 / ssum((D2[i][c] / D2[i][j]) ** (e / 2) for j in range(k)) for c in range(k)])
        T = [[1.0 / (1.0 + (b * D2[i][c] / gam[c]) ** (1.0 / (eta - 1.0))) for c in range(k)] for i in range(n)]
        newV = []
        for c in range(k):
            w = [a * U[i][c] ** m + b * T[i][c] ** eta for i in range(n)]
            sw = ssum(w)
            newV.append([ssum(w[i] * P[i][t] for i in range(n)) / sw for t in range(p)])
        shift = max(math.sqrt(_d2(V[c], newV[c])) for c in range(k))
        V = newV
        if shift < tol:
            break
    lab = [max(range(k), key=lambda c: (U[i][c], -c)) + 1 for i in range(n)]
    return RichResult(
        payload={"centers": V, "membership": U, "typicality": T, "gamma": gam, "cluster": lab, "iter": it}
    )


def growing_neural_gas(
    X,
    *,
    n_signals: int = 5000,
    max_nodes: int = 30,
    eps_b: float = 0.2,
    eps_n: float = 0.006,
    lam: int = 100,
    alpha: float = 0.5,
    a_max: int = 50,
    d: float = 0.995,
    seed: int = 1,
) -> RichResult:
    r"""Growing neural gas (Fritzke 1995); clusters are the connected components of the final graph.

    Each signal (a data point drawn uniformly, Philox stream) moves the
    nearest node by ``eps_b`` and its topological neighbours by ``eps_n``,
    refreshes the edge between the two nearest nodes, ages the winner's
    edges and removes those older than ``a_max`` (and nodes left without
    edges). Every ``lam`` signals a node is inserted halfway between the
    node of largest accumulated error and its worst neighbour; all errors
    decay by ``d``. Each data point takes the component of its nearest node.

    References
    ----------
    Fritzke, B. (1995). A growing neural gas network learns topologies. In
    *Advances in Neural Information Processing Systems 7*, 625-632.

    Examples
    --------
    >>> pts = [[0.1 * (i % 5), 0.1 * (i // 5)] for i in range(25)] + [[9 + 0.1 * (i % 5), 9 + 0.1 * (i // 5)] for i in range(25)]
    >>> r = growing_neural_gas(pts, n_signals=3000, max_nodes=10)
    >>> r.cluster == [1] * 25 + [2] * 25
    True
    """
    P = _pts(X)
    n = len(P)
    u = [float(v) for v in random_uniform(n_signals + 2, seed=seed)]
    W = [list(P[min(int(u[0] * n), n - 1)]), list(P[min(int(u[1] * n), n - 1)])]
    err = [0.0, 0.0]
    edges = {}  # (a, b) with a < b -> age
    for s in range(n_signals):
        x = P[min(int(u[s + 2] * n), n - 1)]
        dd = [_d2(w, x) for w in W]
        order = sorted(range(len(W)), key=lambda j: (dd[j], j))
        s1, s2 = order[0], order[1]
        for e in list(edges):
            if s1 in e:
                edges[e] += 1
        err[s1] += dd[s1]
        W[s1] = [W[s1][t] + eps_b * (x[t] - W[s1][t]) for t in range(len(x))]
        for e in edges:
            if s1 in e:
                o = e[0] if e[1] == s1 else e[1]
                W[o] = [W[o][t] + eps_n * (x[t] - W[o][t]) for t in range(len(x))]
        edges[(min(s1, s2), max(s1, s2))] = 0
        edges = {e: g for e, g in edges.items() if g <= a_max}
        alive = sorted({v for e in edges for v in e})
        if len(alive) < len(W):
            remap = {old: new for new, old in enumerate(alive)}
            W = [W[j] for j in alive]
            err = [err[j] for j in alive]
            edges = {(remap[a], remap[b]): g for (a, b), g in edges.items()}
        if (s + 1) % lam == 0 and len(W) < max_nodes:
            q = max(range(len(W)), key=lambda j: (err[j], -j))
            nb = [e[0] if e[1] == q else e[1] for e in edges if q in e]
            f = max(nb, key=lambda j: (err[j], -j))
            r = len(W)
            W.append([0.5 * (W[q][t] + W[f][t]) for t in range(len(x))])
            del edges[(min(q, f), max(q, f))]
            edges[(min(q, r), max(q, r))] = 0
            edges[(min(f, r), max(f, r))] = 0
            err[q] *= alpha
            err[f] *= alpha
            err.append(err[q])
        err = [v * d for v in err]
    m = len(W)
    par = list(range(m))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for a, b in sorted(edges):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[max(ra, rb)] = min(ra, rb)
    near = [min(range(m), key=lambda j: (_d2(W[j], x), j)) for x in P]
    lab = [find(j) + 1 for j in near]
    return RichResult(
        payload={"cluster": _first_appearance(lab), "nodes": W, "edges": [[a + 1, b + 1] for a, b in sorted(edges)]}
    )


def cheatsheet() -> str:
    return "denclue / flame_clustering / possibilistic_fcm / growing_neural_gas -> density and prototype clustering."
