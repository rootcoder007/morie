# morie.fn -- function file (rootcoder007/morie)
"""Fused Gromov-Wasserstein distance for structured objects."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_fused_gromov_wasserstein"]


def ot_fused_gromov_wasserstein(M, Cx, Cy, a, b, alpha=0.5, max_iter=20):
    """One coupling that has to satisfy the features and the structure at once.

    A labelled graph carries two kinds of information, and the two usual
    distances each throw one away: Wasserstein on the labels ignores the
    edges, Gromov-Wasserstein on the edges ignores the labels.  Fusing
    them forces a single plan to explain both, and ``alpha`` says which
    evidence dominates -- ``alpha = 0`` is plain transport, ``alpha = 1``
    is plain Gromov.  Solved by conditional gradient with an exact
    transport step, so no entropic blur enters the plan.

    Formula: ``min_T (1-alpha) <T, M> + alpha sum_ijkl |Cx_ik - Cy_jl|^2
    T_ij T_kl`` -- Vayer et al. (2020) eq. (3).  The linearised cost is
    ``(1-alpha) M - 4 alpha Cx T Cy`` and the step is ``gamma = 2/(k+2)``.

    Parameters
    ----------
    M : array-like, shape (n, m)
        Feature cost between the two vertex sets.
    Cx : array-like, shape (n, n)
        Structure matrix of the first object, symmetric.
    Cy : array-like, shape (m, m)
        Structure matrix of the second object, symmetric.
    a, b : array-like
        Marginals.
    alpha : float, default 0.5
        Trade-off in [0, 1].
    max_iter : int, default 20
        Conditional-gradient steps.

    Returns
    -------
    RichResult
        ``T``, ``cost``, ``wass_part``, ``gromov_part``, ``n``, ``m``,
        ``iters``.

    References
    ----------
    Vayer, T., Chapel, L., Flamary, R., Tavenard, R. and Courty, N.
    (2020).  Fused Gromov-Wasserstein distance for structured objects.
    Algorithms 13(9):212.  doi:10.3390/a13090212.
    """
    Mm = core.mat(M)
    A = core.mat(Cx)
    B = core.mat(Cy)
    aa = ot.hist(a)
    bb = ot.hist(b)
    n, m = len(aa), len(bb)
    if len(Mm) != n or len(Mm[0]) != m:
        raise ValueError("M must be n by m")
    if len(A) != n or len(B) != m:
        raise ValueError("structure matrices must match the marginals")
    al = float(alpha)
    if al < 0.0 or al > 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    T = [[aa[i] * bb[j] for j in range(m)] for i in range(n)]
    it = int(max_iter)
    for k in range(it):
        _, CT = ot.gw_cost(A, B, T, aa, bb)
        G = [[(1.0 - al) * Mm[i][j] - 4.0 * al * CT[i][j] for j in range(m)]
             for i in range(n)]
        Th, _ = ot.emd(aa, bb, G)
        gam = 2.0 / (k + 2.0)
        T = [[(1.0 - gam) * T[i][j] + gam * Th[i][j] for j in range(m)]
             for i in range(n)]
    gw, _ = ot.gw_cost(A, B, T, aa, bb)
    if gw < 0.0:
        gw = 0.0
    wpart = ot.frob(T, Mm)
    return RichResult(payload={
        "T": T, "cost": (1.0 - al) * wpart + al * gw,
        "wass_part": wpart, "gromov_part": gw, "n": n, "m": m, "iters": it,
        "method": "Fused Gromov-Wasserstein distance"})


def cheatsheet():
    return "otfgw: fused Gromov-Wasserstein distance for structured objects"
