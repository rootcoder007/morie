# morie.fn -- function file (rootcoder007/morie)
"""Bipartite detection, and the spectral symmetry that certifies it.

Source CONSULTED: Cvetkovic, D., Doob, M. & Sachs, H. (1995), *Spectra
of Graphs: Theory and Applications*, 3rd ed., Johann Ambrosius Barth.
A book, not obtainable here; the theorem used is the classical one that
a graph is bipartite if and only if its adjacency spectrum is symmetric
about zero.

The module does NOT run an eigensolver.  The spectral criterion is
tested through its power-sum form, which is exact integer arithmetic on
the adjacency matrix: for a symmetric A with eigenvalues lambda_i,

    trace(A^k) = sum_i lambda_i^k,

so a spectrum symmetric about zero forces trace(A^k) = 0 for every odd
k, and conversely.  Combinatorially trace(A^k) counts closed walks of
length k, so the criterion is "no odd closed walk", i.e. no odd cycle,
i.e. bipartite -- which is why the same answer is also produced here by
breadth-first two-colouring.  Both are computed and returned, and they
must agree; disagreement raises.

Checking odd k up to n is enough: a non-bipartite graph on n vertices
has an odd cycle of length at most n.

The function name in the ledger says "spectral radius bound"; the
module docstring it shipped with says bipartite detection via spectral
symmetry, and that is what the payload keys (``bipartite``,
``evidence``) describe.  The docstring and the keys were followed.
"""

from ._richresult import RichResult

__all__ = ["sgt_spectral_radius_bound"]


def _matmul(X, Y, n):
    return [[sum(X[i][k] * Y[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]


def sgt_spectral_radius_bound(A):
    """Decide bipartiteness, with the odd power-traces as evidence.

    Parameters
    ----------
    A : sequence of sequences, shape (n, n)
        Symmetric adjacency matrix with a zero diagonal.

    Returns
    -------
    RichResult
        Keys ``bipartite``, ``evidence`` (the odd traces
        trace(A^1), trace(A^3), ... up to n), ``max_odd_trace``,
        ``colouring``, ``part_sizes``, ``n_components``, ``n``, ``m``,
        ``method``.
    """
    A = [[float(v) for v in row] for row in A]
    n = len(A)
    if n == 0:
        raise ValueError("empty adjacency matrix")
    for row in A:
        if len(row) != n:
            raise ValueError("adjacency matrix must be square")
    for i in range(n):
        if A[i][i] != 0.0:
            raise ValueError("adjacency matrix must have a zero diagonal")
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > 1e-12 * (1.0 + abs(A[i][j])):
                raise ValueError("adjacency matrix must be symmetric")

    # breadth-first two-colouring
    adj = [[j for j in range(n) if j != i and A[i][j] != 0.0]
           for i in range(n)]
    colour = [0] * n
    ncomp = 0
    combinatorial = True
    for s in range(n):
        if colour[s] != 0:
            continue
        ncomp += 1
        colour[s] = 1
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if colour[v] == 0:
                    colour[v] = -colour[u]
                    queue.append(v)
                elif colour[v] == colour[u]:
                    combinatorial = False

    # odd power traces of A
    evidence = []
    P = [[A[i][j] for j in range(n)] for i in range(n)]  # A^1
    k = 1
    while k <= n:
        evidence.append(sum(P[i][i] for i in range(n)))
        if k + 2 > n:
            break
        P = _matmul(_matmul(P, A, n), A, n)
        k += 2
    max_odd = max(abs(t) for t in evidence)
    spectral = max_odd <= 1e-9

    if spectral != combinatorial:
        raise ValueError(
            "the spectral and combinatorial bipartiteness tests disagree; "
            "this should be impossible for a symmetric 0/1 adjacency "
            "matrix and means the input is not one")

    return RichResult(
        payload={
            "bipartite": combinatorial,
            "evidence": evidence,
            "max_odd_trace": max_odd,
            "colouring": colour,
            "part_sizes": [sum(1 for c in colour if c == 1),
                           sum(1 for c in colour if c == -1)],
            "n_components": ncomp,
            "n": n,
            "m": sum(sum(row) for row in A) / 2.0,
            "method": "bipartite detection; spectrum symmetric about zero "
                      "iff every odd trace(A^k) vanishes",
        }
    )


def cheatsheet():
    return "sgtspr: Bipartite-detection via spectral symmetry"
