# morie.fn -- function file (rootcoder007/morie)
"""EM step (single iteration) for random-effects variance."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["em_step_random_effects"]


def em_step_random_effects(y, X, cluster, sigma2_u, sigma2_e, beta=None):
    """
    EM step (single iteration) for random-effects variance

    Formula: for the random-intercept model y_ij = x_ij' b + u_j + e_ij
    with u_j ~ N(0, sigma2_u) and e_ij ~ N(0, sigma2_e), the conditional
    moments of the E-step are available in closed form,

        uhat_j     = sigma2_u sum_i r_ij / (sigma2_e + n_j sigma2_u)
        var(u_j|y) = sigma2_u sigma2_e / (sigma2_e + n_j sigma2_u)

    with r_ij = y_ij - x_ij' b, and the M-step is

        sigma2_u^(t+1) = (1/J) sum_j [uhat_j^2 + var(u_j | y)]
        sigma2_e^(t+1) = (1/N) [ sum_ij (r_ij - uhat_j)^2
                                 + sum_j n_j var(u_j | y) ]

    ``b`` is the GLS estimate at the current variances, obtained
    cluster-wise from the Sherman-Morrison form of V_j^{-1},

        V_j^{-1} = (1/sigma2_e)(I - sigma2_u J_j / (sigma2_e + n_j sigma2_u))

    or it may be supplied fixed via ``beta``, in which case sigma2_e = 0
    is admissible and the update reduces to its exact limit.

    Parameters
    ----------
    y : array-like
        Response, length N.
    X : array-like
        N x p fixed-effects design (a length-N vector is treated as one
        column; pass an explicit intercept column if you want one).
    cluster : array-like
        Length-N cluster labels.
    sigma2_u : float
        Current between-cluster variance (>= 0).
    sigma2_e : float
        Current residual variance (> 0, or >= 0 when ``beta`` is given).
    beta : array-like, optional
        Fixed-effect vector held fixed instead of re-estimated by GLS.

    Returns
    -------
    result : dict
        Keys: estimate (updated sigma2_u), sigma2_u, sigma2_e, beta,
        u_hat, var_u, J, n, method.

    References
    ----------
    Dempster, Laird & Rubin (1977), JRSS-B 39(1):1-38.
    Laird & Ware (1982), Biometrics 38(4):963-974, doi:10.2307/2529876.
    """
    y = [float(v) for v in y]
    N = len(y)
    if N == 0:
        raise ValueError("empty input: y has no observations")
    Xm = core.mat(X)
    if len(Xm) != N:
        raise ValueError("X must have one row per observation")
    p = len(Xm[0])
    cluster = list(cluster)
    if len(cluster) != N:
        raise ValueError("cluster must have one label per observation")
    s2u = float(sigma2_u)
    s2e = float(sigma2_e)
    if s2u < 0.0:
        raise ValueError("sigma2_u must be non-negative")
    if s2e < 0.0 or (s2e == 0.0 and beta is None):
        raise ValueError("sigma2_e must be positive")
    labs = []
    for c in cluster:
        if c not in labs:
            labs.append(c)
    J = len(labs)
    idx = {c: i for i, c in enumerate(labs)}
    grp = [[] for _ in range(J)]
    for i in range(N):
        grp[idx[cluster[i]]].append(i)
    if beta is None:
        # cluster-wise GLS: X'V^-1 X and X'V^-1 y via Sherman-Morrison
        A = [[0.0] * p for _ in range(p)]
        b = [0.0] * p
        for g in grp:
            nj = len(g)
            f = s2u / (s2e + nj * s2u)
            sx = [sum(Xm[i][a] for i in g) for a in range(p)]
            sy = sum(y[i] for i in g)
            for a in range(p):
                for c in range(p):
                    A[a][c] += (sum(Xm[i][a] * Xm[i][c] for i in g)
                                - f * sx[a] * sx[c]) / s2e
                b[a] += (sum(Xm[i][a] * y[i] for i in g) - f * sx[a] * sy) / s2e
        bet = core.cholsolve(A, b)
    else:
        bet = [float(v) for v in beta]
        if len(bet) != p:
            raise ValueError("beta must have one entry per column of X")
    r = [y[i] - sum(Xm[i][a] * bet[a] for a in range(p)) for i in range(N)]
    uh = [0.0] * J
    vu = [0.0] * J
    for j in range(J):
        g = grp[j]
        nj = len(g)
        den = s2e + nj * s2u
        sr = sum(r[i] for i in g)
        uh[j] = s2u * sr / den if den > 0.0 else 0.0
        vu[j] = s2u * s2e / den if den > 0.0 else 0.0
    new_u = sum(uh[j] * uh[j] + vu[j] for j in range(J)) / J
    tot = 0.0
    for j in range(J):
        for i in grp[j]:
            tot += (r[i] - uh[j]) ** 2
        tot += len(grp[j]) * vu[j]
    new_e = tot / N
    return RichResult(payload={
        "estimate": new_u,
        "sigma2_u": new_u,
        "sigma2_e": new_e,
        "beta": bet,
        "u_hat": uh,
        "var_u": vu,
        "J": J,
        "n": N,
        "method": "EM step (single iteration) for random-effects variance",
    })


def cheatsheet():
    return "emaxr: EM step (single iteration) for random-effects variance"


# compact alias per ledger/NAMING.md
emsteprandomeffects = em_step_random_effects
