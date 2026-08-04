# morie.fn -- function file (rootcoder007/morie)
"""Signless Laplacian Q = D + A of a graph.

Source CONSULTED: Cvetkovic, D., Doob, M. & Sachs, H. (1995), *Spectra
of Graphs: Theory and Applications*, 3rd ed., Johann Ambrosius Barth.
This is a book and could not be obtained; the definition Q = D + A,
with D the diagonal degree matrix, is the standard published one, and
the structural fact returned alongside it is the classical result that

    the multiplicity of 0 as an eigenvalue of Q equals the number of
    bipartite connected components of the graph

which follows directly from the incidence factorisation Q = R R', where
R is the vertex-edge incidence matrix with both endpoints signed +1:
x'Qx = sum over edges (x_u + x_v)^2, so Qx = 0 exactly when x_u = -x_v
across every edge, i.e. when x is a signed 2-colouring of a component.

That multiplicity is computed here COMBINATORIALLY, by two-colouring
each component, not by an eigensolver.  It is exact, and it makes the
Python and R arms agree to the last bit instead of to eigensolver
tolerance.  The parity harness checks it against ``base::eigen`` on Q
and against ``igraph::bipartite_mapping``.
"""

from ._richresult import RichResult

__all__ = ["sgt_signless_laplacian"]


def _components_and_colouring(adj, n):
    """Component labels and a 2-colouring; colour -1 where impossible."""
    comp = [-1] * n
    colour = [0] * n
    bipartite = [True]
    ncomp = 0
    for s in range(n):
        if comp[s] != -1:
            continue
        stack = [s]
        comp[s] = ncomp
        colour[s] = 1
        ok = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = ncomp
                    colour[v] = -colour[u]
                    stack.append(v)
                elif colour[v] == colour[u]:
                    ok = False
        bipartite.append(ok)
        ncomp += 1
    return comp, colour, bipartite[1:], ncomp


def sgt_signless_laplacian(A):
    """Signless Laplacian of an undirected graph.

    Parameters
    ----------
    A : sequence of sequences, shape (n, n)
        Symmetric adjacency matrix with a zero diagonal.  Weights are
        allowed; the degree is the row sum.  Adjacency for the
        combinatorial part is "entry is non-zero".

    Returns
    -------
    RichResult
        Keys ``Q``, ``degree``, ``n``, ``m``, ``trace``,
        ``n_components``, ``bipartite_components``,
        ``zero_eigenvalue_multiplicity``, ``method``.
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
    deg = [sum(A[i]) for i in range(n)]
    Q = [[A[i][j] + (deg[i] if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    adj = [[j for j in range(n) if j != i and A[i][j] != 0.0]
           for i in range(n)]
    _comp, _col, bip, ncomp = _components_and_colouring(adj, n)
    nbip = sum(1 for b in bip if b)
    return RichResult(
        payload={
            "Q": Q,
            "degree": deg,
            "n": n,
            "m": sum(deg) / 2.0,
            "trace": sum(deg),
            "n_components": ncomp,
            "bipartite_components": nbip,
            "zero_eigenvalue_multiplicity": nbip,
            "method": "signless Laplacian Q = D + A",
        }
    )


def cheatsheet():
    return "sgtsig: Signless Laplacian Q = D + A"
