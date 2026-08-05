# morie.fn -- function file (rootcoder007/morie)
"""Random-walk graph kernel between two graphs."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["random_walk_kernel"]


def random_walk_kernel(G1, G2, lam=0.05):
    """Kernel between two graphs via simultaneous walks on both.

    The walks are counted on the DIRECT PRODUCT graph, whose adjacency is
    the Kronecker product ``A1 (x) A2``: a walk there is a pair of walks,
    one in each graph, of equal length.  Summing the discounted counts
    gives the kernel in closed form.  Convergence needs ``lam`` below
    ``1 / (rho(A1) rho(A2))``, which is a much tighter constraint than
    the single-graph case and the usual reason a naive lam of 0.1 blows
    up on dense graphs.

    Formula: ``k(G1, G2) = sum_ij [(I - lam (A1 (x) A2))^{-1}]_ij``.

    Parameters
    ----------
    G1, G2 : array-like, square
        Adjacency matrices; they need not have the same size.
    lam : float, default 0.05
        Discount factor, positive.

    Returns
    -------
    RichResult
        ``estimate`` (the kernel value), ``trace`` of the resolvent,
        ``n1``, ``n2``.

    References
    ----------
    Gaertner, T., Flach, P. & Wrobel, S. (2003).  On graph kernels:
    hardness results and efficient alternatives.  In Learning Theory and
    Kernel Machines (COLT/Kernel 2003), LNCS 2777, pages 129-143.
    doi:10.1007/978-3-540-45167-9_11.
    """
    A = C.mat(G1)
    B = C.mat(G2)
    n1 = len(A)
    n2 = len(B)
    if n1 == 0 or n2 == 0:
        raise ValueError("random_walk_kernel: both graphs must be non-empty")
    for r in A:
        if len(r) != n1:
            raise ValueError("random_walk_kernel: G1 must be square")
    for r in B:
        if len(r) != n2:
            raise ValueError("random_walk_kernel: G2 must be square")
    lam = float(lam)
    if lam <= 0.0:
        raise ValueError("random_walk_kernel: lam must be positive")
    N = n1 * n2
    R = [[0.0] * N for _ in range(N)]
    for i1 in range(n1):
        for i2 in range(n2):
            r = i1 * n2 + i2
            for j1 in range(n1):
                for j2 in range(n2):
                    c = j1 * n2 + j2
                    R[r][c] = (1.0 if r == c else 0.0) - lam * A[i1][j1] * B[i2][j2]
    W = C.inv(R)
    tot = 0.0
    tr = 0.0
    for i in range(N):
        tr += W[i][i]
        for j in range(N):
            tot += W[i][j]
    return RichResult(payload={
        "estimate": tot, "trace": tr, "n1": n1, "n2": n2,
        "method": "Direct-product random-walk graph kernel"})


def cheatsheet():
    return "randW: Random walk graph kernel"
