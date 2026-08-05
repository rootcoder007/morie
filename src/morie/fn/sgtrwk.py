# morie.fn -- function file (rootcoder007/morie)
"""Random-walk kernel of a graph as a resolvent."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sgt_random_walk_kernel"]


def sgt_random_walk_kernel(A, lam=0.1):
    """Geometric series over walk counts, summed in closed form.

    ``sum_k lam^k A^k`` counts walks of every length with a geometric
    discount, and equals ``(I - lam A)^{-1}`` exactly when the series
    converges, i.e. when ``lam`` is below the reciprocal spectral radius.
    Computing the resolvent instead of truncating the series is not an
    optimisation: a truncation at K terms is a different kernel, and is
    not positive definite for the same range of lam.

    Formula: ``K = sum_{k>=0} lam^k A^k = (I - lam A)^{-1}``.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Adjacency matrix.
    lam : float, default 0.1
        Discount; must be positive and small enough for convergence.

    Returns
    -------
    RichResult
        ``K`` (n-by-n), ``estimate`` (the sum of all entries, the kernel
        value used when the whole graph is compared), ``trace``, ``n``.

    References
    ----------
    Gaertner, T., Flach, P. & Wrobel, S. (2003).  On graph kernels:
    hardness results and efficient alternatives.  In Learning Theory and
    Kernel Machines (COLT/Kernel 2003), LNCS 2777, pages 129-143.
    doi:10.1007/978-3-540-45167-9_11.
    """
    M = C.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError("sgt_random_walk_kernel: adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("sgt_random_walk_kernel: adjacency matrix must be square")
    lam = float(lam)
    if lam <= 0.0:
        raise ValueError("sgt_random_walk_kernel: lam must be positive")
    R = [[(1.0 if i == j else 0.0) - lam * M[i][j] for j in range(n)]
         for i in range(n)]
    K = C.inv(R)
    tot = 0.0
    tr = 0.0
    for i in range(n):
        tr += K[i][i]
        for j in range(n):
            tot += K[i][j]
    return RichResult(payload={
        "K": K, "estimate": tot, "trace": tr, "n": n,
        "method": "Random-walk kernel (I - lam A)^{-1}"})


def cheatsheet():
    return "sgtrwk: Random-walk kernel (geometric series)"
