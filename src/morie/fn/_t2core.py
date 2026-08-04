# morie.fn -- function file (rootcoder007/morie)
"""Private numeric helpers for the tail2 batch.  Nothing here is exported.

Every routine runs a FIXED number of steps with no tolerance-based early
exit, so the Python and the R arm of the package execute the same
arithmetic in the same order and agree to machine precision.  A cyclic
Jacobi sweep that stops on a convergence test would stop after a
different number of sweeps on the two arms and silently break parity.
"""

import math

__all__ = []

_JACOBI_SWEEPS = 60


def t2square(a):
    """Coerce to a square list-of-lists of float; return (matrix, n)."""
    m = [[float(v) for v in row] for row in a]
    n = len(m)
    if n == 0 or any(len(row) != n for row in m):
        raise ValueError("expected a non-empty square matrix")
    return m, n


def t2sym(a):
    """Symmetrised copy (A + A^T)/2 of a square matrix."""
    m, n = t2square(a)
    return [[0.5 * (m[i][j] + m[j][i]) for j in range(n)] for i in range(n)], n


def t2eigh(a):
    """Eigenvalues and eigenvectors of a symmetric matrix, cyclic Jacobi.

    60 sweeps, unconditionally.  Returns ``(w, v)`` with ``w`` ascending
    and ``v[i][j]`` the i-th entry of the j-th eigenvector.  Each
    eigenvector is sign-fixed so that its largest-magnitude entry (first
    such index on a tie) is positive, which makes the decomposition a
    function of the input alone.
    """
    m, n = t2sym(a)
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _sweep in range(_JACOBI_SWEEPS):
        for p in range(n - 1):
            for q in range(p + 1, n):
                if m[p][q] == 0.0:
                    continue
                theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
                t = (1.0 if theta >= 0.0 else -1.0) / (
                    abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p] = c * mkp - s * mkq
                    m[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = m[p][k], m[q][k]
                    m[p][k] = c * mpk - s * mqk
                    m[q][k] = s * mpk + c * mqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    vals = [m[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: (vals[i], i))
    w = [vals[i] for i in order]
    vec = [[v[r][order[j]] for j in range(n)] for r in range(n)]
    for j in range(n):
        big = 0
        for i in range(n):
            if abs(vec[i][j]) > abs(vec[big][j]):
                big = i
        if vec[big][j] < 0.0:
            for i in range(n):
                vec[i][j] = -vec[i][j]
    return w, vec


def t2expsym(a):
    """exp(A) for symmetric A, via V diag(exp(w)) V^T."""
    w, v = t2eigh(a)
    n = len(w)
    ew = [math.exp(x) for x in w]
    return [[sum(v[i][k] * ew[k] * v[j][k] for k in range(n))
             for j in range(n)] for i in range(n)]


def t2degree(a):
    """Row sums of a square matrix (the degrees of a weighted graph)."""
    m, n = t2square(a)
    return [sum(m[i]) for i in range(n)], n


def t2adjlist(a):
    """Neighbour index lists, ascending, from a square 0/1 matrix.

    An entry is an edge when it is non-zero; self loops are dropped.
    """
    m, n = t2square(a)
    return [[j for j in range(n) if j != i and m[i][j] != 0.0]
            for i in range(n)], n


def t2brandes(adj):
    """Freeman betweenness by the Brandes (2001) accumulation.

    Unweighted, single-source shortest paths by breadth-first search;
    one BFS per source, then dependencies accumulated in reverse order
    of discovery:  delta[u] += sigma[u]/sigma[v] * (1 + delta[v]).
    Returns the DIRECTED sums, i.e. every ordered pair (s, t) counted
    once; the undirected Freeman score is half of this.
    """
    n = len(adj)
    cb = [0.0] * n
    for src in range(n):
        stack = []
        pred = [[] for _ in range(n)]
        sigma = [0.0] * n
        dist = [-1] * n
        sigma[src] = 1.0
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            stack.append(u)
            for w in adj[u]:
                if dist[w] < 0:
                    dist[w] = dist[u] + 1
                    queue.append(w)
                if dist[w] == dist[u] + 1:
                    sigma[w] = sigma[w] + sigma[u]
                    pred[w].append(u)
        delta = [0.0] * n
        for i in range(len(stack) - 1, -1, -1):
            w = stack[i]
            for u in pred[w]:
                delta[u] = delta[u] + sigma[u] / sigma[w] * (1.0 + delta[w])
            if w != src:
                cb[w] = cb[w] + delta[w]
    return cb


def t2bipartite(adj):
    """Exact two-colouring test; returns (is_bipartite, colour list).

    Colour -1 marks a vertex that no search reached, which cannot occur
    because every component is visited from its lowest-index vertex.
    """
    n = len(adj)
    colour = [-1] * n
    ok = True
    for src in range(n):
        if colour[src] >= 0:
            continue
        colour[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for w in adj[u]:
                if colour[w] < 0:
                    colour[w] = 1 - colour[u]
                    queue.append(w)
                elif colour[w] == colour[u]:
                    ok = False
    return ok, colour
