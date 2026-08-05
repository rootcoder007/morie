# morie.fn -- function file (rootcoder007/morie)
"""Entropic Gromov-Wasserstein coupling."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_gromov_sinkhorn"]


def ot_gromov_sinkhorn(Cx, Cy, a, b, epsilon, max_iter=20, inner_iter=200):
    """Couple two spaces that have no common ground metric.

    Ordinary transport needs a cost between a point of one space and a
    point of the other.  When the two live in different spaces -- a graph
    and a point cloud, say -- no such cost exists, and the only thing that
    can be compared is the pattern of within-space distances.  The
    resulting objective is quadratic in the plan, so it is minimised by
    repeatedly linearising it and running Sinkhorn on the linearisation.

    Formula: ``min_T sum_ijkl |Cx_ik - Cy_jl|^2 T_ij T_kl - eps H(T)``,
    solved by ``T^{l+1} = argmin <T, -Cx T^l Cy> - eps H(T)`` -- Peyre &
    Cuturi (2019) eq. (10.27)-(10.28), p. 176, read from the rendered
    page; Peyre, Cuturi & Solomon (2016).

    Parameters
    ----------
    Cx : array-like, shape (n, n)
        Within-space distances of the first space, symmetric.
    Cy : array-like, shape (m, m)
        Within-space distances of the second space, symmetric.
    a, b : array-like
        Marginals.
    epsilon : float
        Entropic strength, positive.
    max_iter : int, default 20
        Outer linearisations.
    inner_iter : int, default 200
        Sinkhorn sweeps per linearisation.

    Returns
    -------
    RichResult
        ``T``, ``cost`` (the Gromov objective), ``GW`` (its square root),
        ``n``, ``m``, ``iters``.

    References
    ----------
    Peyre, G., Cuturi, M. and Solomon, J. (2016).  Gromov-Wasserstein
    averaging of kernel and distance matrices.  Proceedings of Machine
    Learning Research 48:2664-2672 (ICML).
    """
    A = core.mat(Cx)
    B = core.mat(Cy)
    aa = ot.hist(a)
    bb = ot.hist(b)
    n, m = len(aa), len(bb)
    if len(A) != n or len(A[0]) != n:
        raise ValueError("Cx must be n by n")
    if len(B) != m or len(B[0]) != m:
        raise ValueError("Cy must be m by m")
    eps = float(epsilon)
    T = [[aa[i] * bb[j] for j in range(m)] for i in range(n)]
    it = int(max_iter)
    for _ in range(it):
        _, CT = ot.gw_cost(A, B, T, aa, bb)
        L = [[-CT[i][j] for j in range(m)] for i in range(n)]
        T, _, _ = ot.sinkhorn(aa, bb, L, eps, inner_iter)
    cost, _ = ot.gw_cost(A, B, T, aa, bb)
    if cost < 0.0:
        cost = 0.0
    return RichResult(payload={
        "T": T, "cost": cost, "GW": cost ** 0.5, "n": n, "m": m,
        "iters": it,
        "method": "Entropic Gromov-Wasserstein coupling"})


def cheatsheet():
    return "otgws: entropic Gromov-Wasserstein coupling of two metric spaces"
