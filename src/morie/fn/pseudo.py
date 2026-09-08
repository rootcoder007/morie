# SPDX-License-Identifier: AGPL-3.0-or-later
"""Path-specific effects in linear structural equation models."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["pseudo", "path_specific_effect"]


def _path_sum(B, n):
    # I + B + B^2 + ... ; terminates because B is nilpotent on a DAG.
    T = np.eye(n)
    P = np.eye(n)
    for _ in range(n):
        P = P @ B
        T = T + P
    return T


def pseudo(B, x, y, edges=None):
    """
    Path-specific effect of variable x on variable y in a linear
    structural equation model.

    Avin, Shpitser and Pearl (2005) define the g-specific effect as
    the effect transmitted along a chosen subgraph g of edges, with
    every edge outside g turned off (held at its reference behaviour);
    Pearl (2001, Section 2) notes that in linear systems these
    path-restricted effects are the sums of products of the path
    coefficients along the retained paths, the classical path-analysis
    rule. For a linear SEM with coefficient matrix B (B[i, j] is the
    structural coefficient on the edge i -> j, acyclic), the effect of
    a unit change of x on y along the subgraph g is therefore

        PSE_g(x -> y) = [(I - B_g)^-1]_{x, y}
                      = sum over directed x -> y paths inside g of
                        the product of edge coefficients,

    where B_g zeroes every edge not in g. With g the full graph this
    is the total effect; with g = {x -> y} the (natural = controlled,
    in linear models) direct effect; total minus direct is the
    indirect effect. In linear models natural and controlled
    path-specific effects coincide (Pearl 2001, Section 2; effects are
    independent of the values at which intermediate variables are
    held).

    Parameters
    ----------
    B : array-like, shape (k, k)
        Structural coefficient matrix of an acyclic model,
        B[i, j] = coefficient of the edge i -> j.
    x, y : int
        0-based indices of the cause and outcome variables.
    edges : list of (i, j) pairs, optional
        The retained path set g. Default: all edges (total effect).

    Returns
    -------
    result : RichResult
        Keys: estimate (the g-specific effect), total (full-graph
        effect), direct (single-edge x -> y effect), indirect
        (total - direct), n_edges_used.

    References
    ----------
    Avin, C., Shpitser, I. and Pearl, J. (2005), "Identifiability of
    path-specific effects", Proc. 19th IJCAI, 357-363; edge-subgraph
    definition of path-specific effects (Section 2). Local copy:
    fetched-wave3/avin-shpitser-pearl-2005-path-specific-effects-IJCAI.pdf
    Pearl, J. (2001), "Direct and indirect effects", Proc. 17th UAI,
    411-420; Section 2 (path coefficients in linear models, natural
    effects). Local copy:
    fetched-wave3/pearl-2001-direct-indirect-effects-UAI.pdf
    """
    Bm = np.asarray(B, dtype=float)
    if Bm.ndim != 2 or Bm.shape[0] != Bm.shape[1]:
        raise ValueError("B must be a square matrix")
    k = Bm.shape[0]
    x = int(x)
    y = int(y)
    if not (0 <= x < k and 0 <= y < k) or x == y:
        raise ValueError("x and y must be distinct indices into B")
    # acyclicity: B must be nilpotent
    P = np.eye(k)
    for _ in range(k):
        P = P @ Bm
    if float(np.max(np.abs(P))) > 0.0:
        raise ValueError("B is not acyclic (B^k != 0)")
    if edges is None:
        Bg = Bm
        used = int(sum(1 for i in range(k) for j in range(k)
                       if float(Bm[i, j]) != 0.0))
    else:
        Bg = np.zeros((k, k))
        used = 0
        for (i, j) in edges:
            i = int(i)
            j = int(j)
            if not (0 <= i < k and 0 <= j < k):
                raise ValueError("edge (%d, %d) out of range" % (i, j))
            Bg[i, j] = Bm[i, j]
            used += 1
    T_g = _path_sum(Bg, k)
    T_full = _path_sum(Bm, k)
    direct = float(Bm[x, y])
    total = float(T_full[x, y])
    return RichResult(payload={
        "estimate": float(T_g[x, y]),
        "total": total,
        "direct": direct,
        "indirect": total - direct,
        "n_edges_used": used,
        "method": "Avin-Shpitser-Pearl (2005) path-specific effect, linear path rule",
    })


path_specific_effect = pseudo


def cheatsheet():
    return "pseudo(B, x, y, edges) -> path-specific effect along the retained edges of a linear SEM."
