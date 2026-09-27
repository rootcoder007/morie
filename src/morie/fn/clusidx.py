"""Spatial clustering indices ACL, RCL and SP (White 1983; Massey and Denton 1988).

White, M. J. (1983). The measurement of spatial segregation. American Journal of Sociology
88, 1008-1018. Massey, D. S. and Denton, N. A. (1988). Social Forces 67, 281-315.
"""

from ._richresult import RichResult
from ._segcore import col_totals, counts, decay, matvec, row_totals, ssum

__all__ = ["clustering_index"]


def clustering_index(x, contiguity=None, distance=None, beta=1.0):
    r"""Clustering: whether a group's units adjoin one another.

    ACL_k = (sum_i x_ik/X_k sum_j c_ij x_jk - X_k sum_ij c_ij / n^2) / (sum_i x_ik/X_k sum_j c_ij t_j - X_k sum_ij c_ij / n^2)
    with c the contiguity matrix with unit diagonal (Massey and Denton 1988), or c_ij =
    exp(-beta d_ij) from ``distance``. With distances, P_kk = sum_ij x_ik x_jk exp(-beta d_ij) / X_k^2,
    the relative clustering RCL_kl = P_kk / P_ll - 1 and White's spatial proximity
    SP = sum_k X_k P_kk / (N P_tt), P_tt the same average proximity for the whole population.

    Parameters
    ----------
    x : units x groups counts
    contiguity : n x n 0/1 matrix, optional (its diagonal is set to 1)
    distance : n x n matrix, optional
    beta : float

    Returns
    -------
    RichResult
        Keys: ACL, and with ``distance`` also RCL (groups x groups) and SP.

    References
    ----------
    Massey, D. S. and Denton, N. A. (1988). Social Forces 67, 281-315.
    White, M. J. (1983). American Journal of Sociology 88, 1008-1018.

    Examples
    --------
    >>> r = clustering_index([[4, 1], [3, 2], [1, 4], [0, 5]], contiguity=[[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    >>> r["ACL"][0] > r["ACL"][1] - 1
    True
    """
    if contiguity is None and distance is None:
        raise ValueError("give contiguity or distance")
    keep = [i for i, r in enumerate(x) if ssum(float(v) for v in r) > 0]
    X = counts(x)
    n, g = len(X), len(X[0])
    tot, t = col_totals(X), row_totals(X)
    if contiguity is not None:
        c = [[float(contiguity[i][j]) for j in keep] for i in keep]
        for i in range(n):
            c[i][i] = 1.0
    else:
        c = decay([[float(distance[i][j]) for j in keep] for i in keep], float(beta))
    csum = ssum(ssum(r) for r in c)
    ct = matvec(c, t)
    ACL = []
    for k in range(g):
        xk = [r[k] for r in X]
        cx = matvec(c, xk)
        v1 = ssum(cx[i] * xk[i] / tot[k] for i in range(n))
        v2 = csum * tot[k] / n**2
        v3 = ssum(ct[i] * xk[i] / tot[k] for i in range(n))
        ACL.append((v1 - v2) / (v3 - v2))
    out = {"ACL": ACL}
    if distance is not None:
        E = decay([[float(distance[i][j]) for j in keep] for i in keep], float(beta))
        P = []
        for k in range(g):
            xk = [r[k] for r in X]
            ex = matvec(E, xk)
            P.append(ssum(ex[i] * xk[i] for i in range(n)) / tot[k] ** 2)
        et = matvec(E, t)
        Ptt = ssum(et[i] * t[i] for i in range(n)) / ssum(t) ** 2
        out["RCL"] = [[P[a] / P[b] - 1 for b in range(g)] for a in range(g)]
        out["SP"] = ssum(P[k] * tot[k] for k in range(g)) / (ssum(t) * Ptt)
    return RichResult(title="Clustering indices", summary_lines=[("groups", g)], payload=out)


def cheatsheet():
    return "clusidx: Massey-Denton clustering ACL, RCL and White's spatial proximity SP"
