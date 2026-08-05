# morie.fn -- function file (rootcoder007/morie)
"""Free-support Wasserstein barycenter."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_barycenter_free"]


def _bary_step(Y, clouds, w):
    n = len(Y)
    d = len(Y[0])
    a = [1.0 / n] * n
    Z = [[0.0] * d for _ in range(n)]
    cost = 0.0
    for k, Xk in enumerate(clouds):
        mk = len(Xk)
        C = ot.costmat(Y, Xk, 2)
        T, c = ot.emd(a, [1.0 / mk] * mk, C)
        cost += w[k] * c
        for i in range(n):
            for j in range(mk):
                if T[i][j] == 0.0:
                    continue
                for t in range(d):
                    Z[i][t] += w[k] * n * T[i][j] * Xk[j][t]
    return Z, cost


def ot_barycenter_free(X_list, weights, n_supp, max_iter=20):
    """Barycentre whose support is found rather than fixed in advance.

    A fixed grid forces the barycentre onto points that may all be far
    from the data, which in more than two dimensions is fatal.  Letting
    the atoms move turns the problem into an alternating scheme -- solve
    the transports, then move each atom to the weighted average of the
    mass it received -- which is Lloyd's algorithm with a transport plan
    in place of the nearest-point assignment.

    Formula: alternate ``T_k = argmin <T, C(Y, X_k)>`` and
    ``Y <- n sum_k w_k T_k X_k`` -- Cuturi & Doucet (2014) Section 4.
    Uniform weights are kept on the support throughout.

    Parameters
    ----------
    X_list : sequence of arrays, each (n_k, d)
        Input point clouds.
    weights : array-like
        Barycentric weights; rescaled to sum to one.
    n_supp : int
        Number of support atoms.
    max_iter : int, default 20
        Alternations.

    Returns
    -------
    RichResult
        ``Y``, ``weights_y``, ``cost``, ``n_supp``, ``d``, ``K``,
        ``iters``.

    References
    ----------
    Cuturi, M. and Doucet, A. (2014).  Fast computation of Wasserstein
    barycenters.  Proceedings of Machine Learning Research 32:685-693
    (ICML).
    """
    clouds = [core.mat(X) for X in X_list]
    K = len(clouds)
    if K == 0:
        raise ValueError("no input clouds")
    d = len(clouds[0][0])
    for Xk in clouds:
        if len(Xk[0]) != d:
            raise ValueError("all clouds must share a dimension")
    w = ot.hist(weights, normalise=True)
    if len(w) != K:
        raise ValueError("one weight per cloud is required")
    ns = int(n_supp)
    pool = [row for Xk in clouds for row in Xk]
    if ns < 1 or ns > len(pool):
        raise ValueError("n_supp must lie between 1 and the pooled size")
    step = len(pool) / float(ns)
    Y = [list(pool[int(k * step)]) for k in range(ns)]
    it = int(max_iter)
    cost = 0.0
    for _ in range(it):
        Y, cost = _bary_step(Y, clouds, w)
    return RichResult(payload={
        "Y": Y, "weights_y": [1.0 / ns] * ns, "cost": cost,
        "n_supp": ns, "d": d, "K": K, "iters": it,
        "method": "Free-support Wasserstein barycenter"})


def cheatsheet():
    return "otbarfree: free-support Wasserstein barycenter of point clouds"
