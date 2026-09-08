# morie.fn -- function file (rootcoder007/morie)
"""Entropic Wasserstein barycenter on a fixed support."""

import math

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_barycenter_fixed"]


def ot_barycenter_fixed(A, C_list, weights, epsilon, max_iter=200):
    """Average measures the way transport says they should be averaged.

    The Euclidean average of two shifted bumps is two bumps; the
    Wasserstein average is one bump in between.  That is the whole point:
    the barycenter respects the geometry of the ground space rather than
    the vector-space structure of the histograms.  Fixing the support
    turns the problem into a coupled set of Sinkhorn problems, one per
    input, sharing a common row scaling.

    Formula: ``argmin_nu sum_k w_k OT_eps(mu_k, nu)``, solved by the
    iterations ``u_k = nu/(K_k v_k)``, ``nu = prod_k (K_k v_k)^{w_k}``,
    ``v_k = mu_k/(K_k' u_k)`` -- Benamou et al. (2015) Section 3.2;
    Peyre & Cuturi (2019) eq. (9.11), (9.15).

    Parameters
    ----------
    A : array-like, shape (n, K)
        Input histograms, one per column, all on the barycentre's support.
    C_list : sequence of K arrays, each (n, n)
        Ground cost between the barycentre support and each input.
    weights : array-like, shape (K,)
        Barycentric weights; rescaled to sum to one.
    epsilon : float
        Entropic strength, positive.
    max_iter : int, default 200
        Sweeps.

    Returns
    -------
    RichResult
        ``bary``, ``mass``, ``n``, ``K``, ``iters``.

    References
    ----------
    Benamou, J.-D., Carlier, G., Cuturi, M., Nenna, L. and Peyre, G.
    (2015).  Iterative Bregman projections for regularized transportation
    problems.  SIAM Journal on Scientific Computing 37(2):A1111-A1138.
    doi:10.1137/141000439.  Cuturi, M. and Doucet, A. (2014).  Fast
    computation of Wasserstein barycenters.  Proceedings of Machine
    Learning Research 32:685-693 (ICML).
    """
    Am = core.mat(A)
    n = len(Am)
    K = len(Am[0])
    Cs = [core.mat(c) for c in C_list]
    if len(Cs) != K:
        raise ValueError("one cost matrix per input histogram is required")
    for c in Cs:
        if len(c) != n or len(c[0]) != n:
            raise ValueError("each cost matrix must be n by n")
    w = ot.hist(weights, normalise=True)
    if len(w) != K:
        raise ValueError("one weight per input histogram is required")
    eps = float(epsilon)
    if eps <= 0.0:
        raise ValueError("epsilon must be positive")
    Ks = [[[math.exp(-c[i][j] / eps) for j in range(n)] for i in range(n)]
          for c in Cs]
    v = [[1.0] * n for _ in range(K)]
    bary = [1.0 / n] * n
    it = int(max_iter)
    for _ in range(it):
        Kv = [[sum(Ks[k][i][j] * v[k][j] for j in range(n)) for i in range(n)]
              for k in range(K)]
        bary = []
        for i in range(n):
            s = 0.0
            for k in range(K):
                s += w[k] * (math.log(Kv[k][i]) if Kv[k][i] > 0.0
                             else float("-inf"))
            bary.append(math.exp(s) if s > float("-inf") else 0.0)
        u = [[bary[i] / Kv[k][i] if Kv[k][i] > 0.0 else 0.0 for i in range(n)]
             for k in range(K)]
        for k in range(K):
            for j in range(n):
                s = sum(u[k][i] * Ks[k][i][j] for i in range(n))
                v[k][j] = Am[j][k] / s if s > 0.0 else 0.0
    return RichResult(payload={
        "bary": bary, "mass": sum(bary), "n": n, "K": K, "iters": it,
        "method": "Entropic Wasserstein barycenter, fixed support"})


def cheatsheet():
    return "otbar: entropic Wasserstein barycenter on a fixed support"
