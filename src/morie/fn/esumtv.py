# morie.fn -- function file (rootcoder007/morie)
"""Effective resistance between nodes."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["effective_resistance"]


def effective_resistance(G, u, v):
    """
    Effective resistance between nodes

    Formula: R_uv = (e_u - e_v)' L^+ (e_u - e_v), the resistance distance
    of Klein & Randic (1993), where L = D - A is the graph Laplacian and
    L^+ its Moore-Penrose inverse.

    Rather than forming L^+ explicitly the linear system is grounded:
    L is singular with the all-ones vector in its kernel, so deleting
    row/column ``v`` gives a nonsingular matrix, and the solution x of
    the grounded system against e_u satisfies R_uv = x_u.  This is exact
    and avoids a pseudoinverse entirely.

    Parameters
    ----------
    G : array-like
        Symmetric non-negative weight (or 0/1 adjacency) matrix.  Edge
        weights are read as CONDUCTANCES, per Klein & Randic.
    u, v : int
        Zero-based node indices; they must lie in the same component.

    Returns
    -------
    result : dict
        Keys: estimate (R_uv), resistance, degree_u, degree_v, n, method.

    References
    ----------
    Klein & Randic (1993), Journal of Mathematical Chemistry 12(1):81-95,
    doi:10.1007/BF01164627.
    """
    A = core.mat(G)
    n = len(A)
    if n == 0:
        raise ValueError("empty input: G has no nodes")
    if any(len(r) != n for r in A):
        raise ValueError("G must be square")
    for i in range(n):
        for j in range(n):
            if A[i][j] < 0.0:
                raise ValueError("weights must be non-negative")
            if abs(A[i][j] - A[j][i]) > 1e-12:
                raise ValueError("G must be symmetric")
    u = int(u)
    v = int(v)
    if not (0 <= u < n) or not (0 <= v < n):
        raise ValueError("u and v must be valid node indices")
    if u == v:
        return RichResult(payload={
            "estimate": 0.0, "resistance": 0.0,
            "degree_u": sum(A[u]) - A[u][u], "degree_v": sum(A[v]) - A[v][v],
            "n": n, "method": "Effective resistance between nodes"})
    deg = [sum(A[i]) - A[i][i] for i in range(n)]
    L = [[(deg[i] if i == j else 0.0) - (A[i][j] if i != j else 0.0)
          for j in range(n)] for i in range(n)]
    keep = [i for i in range(n) if i != v]
    Lg = [[L[i][j] for j in keep] for i in keep]
    b = [1.0 if i == u else 0.0 for i in keep]
    # grounded Laplacian is symmetric positive definite on a connected
    # component; a chol failure means u and v are in different components
    try:
        x = core.cholsolve(Lg, b)
    except Exception:
        raise ValueError("u and v are not connected")
    R = x[keep.index(u)]
    return RichResult(payload={
        "estimate": R,
        "resistance": R,
        "degree_u": deg[u],
        "degree_v": deg[v],
        "n": n,
        "method": "Effective resistance between nodes",
    })


def cheatsheet():
    return "esumtv: Effective resistance between nodes"


# compact alias per ledger/NAMING.md
effectiveresistance = effective_resistance
