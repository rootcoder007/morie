"""Functional manifold learning: geodesic embedding of curve data.

Functional data often lives on a low-dimensional manifold inside the
infinite-dimensional space of curves. A family of growth curves differing
only in the timing of a growth spurt is a ONE-parameter family, but the
straight-line distance between two such curves says almost nothing about
how far apart their timings are: shift a peak far enough and the L2
distance saturates, because the curves stop overlapping at all and
moving further apart cannot make them any more different.

That saturation is why linear methods mislead here. Principal components
of such data produce two or three components with no interpretation,
because a curved one-dimensional set needs several linear directions to
be covered. The fix is to measure distance ALONG the set instead of
through the ambient space: build a neighbourhood graph, take shortest
paths through it as geodesic distances, and embed those. Near neighbours
are where the straight-line distance is trustworthy, and the graph
stitches those local truths into a global one.

Three routes, and the difference between them is the whole subject:

  "mds"     classical scaling on the raw L2 distances. Linear, and the
            thing the manifold view is correcting.
  "isomap"  the same scaling applied to GEODESIC distances from the
            neighbourhood graph.
  "geodesic_only"  stops after the graph, for a caller who wants the
            distances rather than an embedding.

The eigendecomposition is a Jacobi sweep written out here rather than a
library call. That is deliberate: eigenvectors are only defined up to
sign and up to rotation within a repeated eigenvalue, so two library
routines can both be correct and disagree, and a module whose output
flips sign depending on which linear algebra it was linked against is
not reproducible. The sweep count is fixed, the ordering is by
descending eigenvalue with ties broken by index, and each vector is
signed so its largest-magnitude entry is positive.

A neighbourhood graph that is not connected is the failure this method
actually hits, and it is reported rather than patched. If k is too small
the graph falls into pieces, the geodesic between pieces is infinite,
and there is no embedding -- the honest response is to say so and name
how many components there were, not to substitute a large finite number
and produce coordinates that mean nothing.

References
  Chen, D. and Mueller, H.-G. (2012) "Nonlinear manifold representations
    for functional data." The Annals of Statistics 40(1), 1-29.
    doi:10.1214/11-AOS936. Manifold representations for curve data.
  Tenenbaum, J.B., de Silva, V. and Langford, J.C. (2000) "A global
    geometric framework for nonlinear dimensionality reduction."
    Science 290(5500), 2319-2323. Isomap: the neighbourhood graph,
    shortest-path geodesics and classical scaling.
  Torgerson, W.S. (1952) "Multidimensional scaling: I. Theory and
    method." Psychometrika 17(4), 401-419. Classical scaling.
  Floyd, R.W. (1962) "Algorithm 97: shortest path." Communications of
    the ACM 5(6), 345. The all-pairs shortest path used here.
  Jacobi, C.G.J. (1846) "Ueber ein leichtes Verfahren die in der Theorie
    der Saecularstoerungen vorkommenden Gleichungen numerisch
    aufzuloesen." Journal fuer die reine und angewandte Mathematik 30,
    51-94. The eigenvalue sweep.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["manfd", "manifold_functional", "l2_distances", "knn_graph",
           "shortest_paths", "jacobi_eigen", "classical_scaling",
           "METHODS", "cheatsheet"]

METHODS = ("isomap", "mds", "geodesic_only")

_INF = float("inf")


def l2_distances(Y, grid=None):
    """Pairwise L2 distance between curves, by the trapezoid rule.

    The trapezoid rather than a plain sum of squares: functional data
    is a sample of a function, and the distance between two functions
    is an integral. On an evenly spaced grid the two differ by a
    constant factor, but on an uneven one they differ by more than
    that, and the constant is not the point.
    """
    n = len(Y)
    p = len(Y[0])
    if any(len(r) != p for r in Y):
        raise ValueError("every curve must be sampled on the same grid")
    if grid is None:
        grid = [float(t) for t in range(p)]
    grid = [float(t) for t in grid]
    if len(grid) != p:
        raise ValueError("the grid must match the curve length")
    for t in range(p - 1):
        if grid[t + 1] <= grid[t]:
            raise ValueError("the grid must be strictly increasing")
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            terms = []
            for t in range(p - 1):
                a = Y[i][t] - Y[j][t]
                b = Y[i][t + 1] - Y[j][t + 1]
                terms.append(0.5 * (a * a + b * b)
                             * (grid[t + 1] - grid[t]))
            v = math.sqrt(_w.csum(terms)) if terms else 0.0
            D[i][j] = v
            D[j][i] = v
    return D


def knn_graph(D, k, symmetric=True):
    """The k-nearest-neighbour graph, as a weighted adjacency matrix.

    Symmetrised by union: i and j are joined if EITHER lists the other.
    Without that the graph is directed and the shortest path between two
    points can depend on which way you walk it, which a distance may
    not do.
    """
    n = len(D)
    k = int(k)
    if k < 1:
        raise ValueError("each point needs at least one neighbour")
    if k >= n:
        raise ValueError("k must be smaller than the sample size")
    A = [[_INF] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = 0.0
        order = sorted(range(n), key=lambda j: (D[i][j], j))
        taken = 0
        for j in order:
            if j == i:
                continue
            A[i][j] = D[i][j]
            taken += 1
            if taken >= k:
                break
    if symmetric:
        for i in range(n):
            for j in range(n):
                if A[j][i] < A[i][j]:
                    A[i][j] = A[j][i]
    return A


def shortest_paths(A):
    """All-pairs shortest paths by Floyd-Warshall, and the components.

    Returns the geodesic matrix and the number of connected components.
    An unreachable pair keeps an infinite distance, which is the truth
    about it.
    """
    n = len(A)
    G = [list(row) for row in A]
    for m in range(n):
        for i in range(n):
            if G[i][m] == _INF:
                continue
            for j in range(n):
                if G[m][j] == _INF:
                    continue
                v = G[i][m] + G[m][j]
                if v < G[i][j]:
                    G[i][j] = v
    seen = [False] * n
    comp = 0
    for i in range(n):
        if seen[i]:
            continue
        comp += 1
        for j in range(n):
            if G[i][j] < _INF:
                seen[j] = True
    return G, comp


def jacobi_eigen(A, sweeps=60):
    """Symmetric eigendecomposition by cyclic Jacobi rotations.

    Written out rather than delegated because eigenvectors are defined
    only up to sign, and up to rotation inside a repeated eigenvalue, so
    two correct library routines can disagree. A fixed number of sweeps,
    a fixed ordering and a fixed sign convention make the answer a
    function of the matrix alone.

    Returns eigenvalues in descending order and their vectors as
    columns, each signed so its largest-magnitude entry is positive.
    """
    n = len(A)
    a = [list(row) for row in A]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(int(sweeps)):
        off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off += a[i][j] * a[i][j]
        if off <= 1e-30:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0.0 else -1.0) / (
                    abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for r in range(n):
                    arp = a[r][p]
                    arq = a[r][q]
                    a[r][p] = c * arp - s * arq
                    a[r][q] = s * arp + c * arq
                for r in range(n):
                    apr = a[p][r]
                    aqr = a[q][r]
                    a[p][r] = c * apr - s * aqr
                    a[q][r] = s * apr + c * aqr
                for r in range(n):
                    vrp = v[r][p]
                    vrq = v[r][q]
                    v[r][p] = c * vrp - s * vrq
                    v[r][q] = s * vrp + c * vrq
    vals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: (-vals[i], i))
    ev = [vals[i] for i in order]
    vec = [[v[r][i] for i in order] for r in range(n)]
    # Sign convention: the largest-magnitude entry of each vector is
    # made positive. Without it the embedding could come out mirrored
    # for no reason a reader could see.
    for j in range(n):
        best = 0
        for r in range(n):
            if abs(vec[r][j]) > abs(vec[best][j]):
                best = r
        if vec[best][j] < 0.0:
            for r in range(n):
                vec[r][j] = -vec[r][j]
    return ev, vec


def classical_scaling(D, dim=2, sweeps=60):
    """Torgerson scaling: double-centre the squared distances, embed.

    B = -1/2 J D^2 J with J the centring matrix. The eigenvectors of B
    scaled by the square roots of its eigenvalues are the coordinates.
    A negative eigenvalue means the distances were not Euclidean, which
    happens routinely with geodesics and is reported rather than
    silently clipped away.
    """
    n = len(D)
    d2 = [[D[i][j] * D[i][j] for j in range(n)] for i in range(n)]
    rmean = [_w.csum(d2[i]) / n for i in range(n)]
    cmean = [_w.csum(d2[i][j] for i in range(n)) / n for j in range(n)]
    gmean = _w.csum(rmean) / n
    B = [[-0.5 * (d2[i][j] - rmean[i] - cmean[j] + gmean)
          for j in range(n)] for i in range(n)]
    ev, vec = jacobi_eigen(B, sweeps)
    dim = int(dim)
    if dim < 1 or dim > n:
        raise ValueError("the embedding dimension must lie in 1..n")
    coords = []
    for i in range(n):
        row = []
        for j in range(dim):
            lam = ev[j]
            row.append(vec[i][j] * math.sqrt(lam) if lam > 0.0 else 0.0)
        coords.append(row)
    n_neg = sum(1 for v in ev if v < -1e-9)
    return coords, ev, n_neg, B


def manifold_functional(Y, k=4, method="isomap", grid=None, dim=2,
                        sweeps=60):
    """Embed functional data on its manifold.

    Parameters
    ----------
    Y : sequence of sequences
        One curve per row, all sampled on the same grid.
    k : int
        Neighbours per point in the graph.
    method : str
        A member of METHODS.
    grid : sequence or None
        The sampling points. Integers 0..p-1 when omitted.
    dim : int
        Embedding dimension.

    Returns
    -------
    RichResult
        The coordinates, the eigenvalues, the geodesic distances, the
        number of connected components and the residual variance of the
        embedding.

    References
    ----------
    Chen and Mueller (2012) Ann Statist 40(1), 1-29; Tenenbaum, de Silva
    and Langford (2000) Science 290(5500), 2319-2323.
    """
    if method not in METHODS:
        raise ValueError("method must be one of %r" % (METHODS,))
    ys = [[float(v) for v in row] for row in Y]
    n = len(ys)
    if n < 3:
        raise ValueError("need at least three curves")
    D = l2_distances(ys, grid)
    A = knn_graph(D, k)
    G, comp = shortest_paths(A)
    disconnected = comp > 1

    if method == "geodesic_only":
        coords = []
        ev = []
        n_neg = 0
    elif disconnected:
        # No embedding exists: some pairs are infinitely far apart in
        # the graph, and substituting a big finite number would invent
        # a geometry the data never had.
        coords = []
        ev = []
        n_neg = 0
    else:
        src = G if method == "isomap" else D
        coords, ev, n_neg, _ = classical_scaling(src, dim, sweeps)

    # Residual variance: one minus the squared correlation between the
    # input distances and the distances in the embedding. It is the
    # standard isomap diagnostic and it says how much of the geometry
    # survived.
    resid = float("nan")
    if coords:
        src = G if method == "isomap" else D
        a = []
        b = []
        for i in range(n):
            for j in range(i + 1, n):
                a.append(src[i][j])
                e = math.sqrt(_w.csum(
                    (coords[i][t] - coords[j][t])
                    * (coords[i][t] - coords[j][t])
                    for t in range(len(coords[i]))))
                b.append(e)
        ma = _w.csum(a) / len(a)
        mb = _w.csum(b) / len(b)
        saa = _w.csum((v - ma) * (v - ma) for v in a)
        sbb = _w.csum((v - mb) * (v - mb) for v in b)
        sab = _w.csum((a[t] - ma) * (b[t] - mb) for t in range(len(a)))
        if saa > 0.0 and sbb > 0.0:
            r = sab / math.sqrt(saa * sbb)
            resid = 1.0 - r * r

    finite = [G[i][j] for i in range(n) for j in range(n)
              if G[i][j] < _INF]
    return RichResult(payload={
        "coords": coords,
        "eigenvalues": ev,
        "distance": D,
        "geodesic": G,
        "n_components": comp,
        "disconnected": disconnected,
        "n_negative_eigenvalues": n_neg,
        "residual_variance": resid,
        "estimate": resid,
        "se": float("nan"),
        "geodesic_max": max(finite) if finite else float("nan"),
        "n": n,
        "k": int(k),
        "dim": int(dim),
        "method": method,
        "name": "functional manifold representation",
    })


manfd = manifold_functional


def cheatsheet():
    return ("manfd: functional manifold learning. methods "
            + ", ".join(METHODS)
            + "; L2 curve distances, k-NN graph geodesics, Torgerson "
              "scaling on a written-out Jacobi eigensolver")
