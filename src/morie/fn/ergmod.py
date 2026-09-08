# morie.fn -- function file (rootcoder007/morie)
"""Exponential random graph model."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["ergm"]

_SUPPORTED = ("edges", "twostar", "triangle")


def _change_stats(A, i, j, names):
    """Change statistics for toggling dyad (i, j) on an undirected graph."""
    n = len(A)
    out = []
    for nm in names:
        if nm == "edges":
            out.append(1.0)
        elif nm == "twostar":
            di = sum(A[i]) - A[i][j]
            dj = sum(A[j]) - A[i][j]
            out.append(float(di + dj))
        else:  # triangle
            out.append(float(sum(1 for k in range(n)
                                 if k != i and k != j and A[i][k] and A[j][k])))
    return out


def ergm(G, statistics=("edges",), theta_init=None, iters=100, tol=1e-11):
    """
    Exponential random graph model

    Formula: P(G = g) = exp(theta' s(g)) / kappa(theta), the Markov graph
    family of Frank & Strauss (1986).

    Fitted here by MAXIMUM PSEUDO-LIKELIHOOD.  The normalising constant
    kappa is intractable, so the full likelihood needs MCMC; the
    pseudo-likelihood replaces it with the product of the conditional
    dyad probabilities, each of which is exactly logistic in the CHANGE
    STATISTICS delta_ij = s(g with edge ij) - s(g without),

        logit P(A_ij = 1 | rest) = theta' delta_ij

    so the fit is an ordinary logistic regression over the C(n,2) dyads,
    solved by Newton-Raphson.  This is deterministic and exact -- no
    sampler, so both language arms land on identical numbers -- and it is
    the standard starting value for the MCMC-MLE of Hunter & Handcock
    (2006).  It is NOT the MLE; for dependent terms (twostar, triangle)
    the pseudo-likelihood estimate is biased, which is precisely why
    Hunter & Handcock's paper exists.

    Parameters
    ----------
    G : array-like
        Symmetric 0/1 adjacency matrix, zero diagonal.
    statistics : sequence of str
        Any of "edges", "twostar", "triangle".
    theta_init : array-like, optional
        Starting values for Newton-Raphson (default zeros).
    iters : int
        Maximum Newton steps.
    tol : float
        Convergence tolerance on the coefficient change.

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), theta, se, observed_stats,
        pseudo_loglik, n_dyads, iters_used, n, method.

    References
    ----------
    Frank & Strauss (1986), JASA 81(395):832-842, doi:10.2307/2289017.
    Hunter & Handcock (2006), Journal of Computational and Graphical
    Statistics 15(3):565-583, doi:10.1198/106186006X133069.
    """
    A = core.mat(G)
    n = len(A)
    if n == 0:
        raise ValueError("empty input: G has no nodes")
    if any(len(r) != n for r in A):
        raise ValueError("G must be square")
    for i in range(n):
        if A[i][i] != 0.0:
            raise ValueError("G must have a zero diagonal")
        for j in range(n):
            if A[i][j] not in (0.0, 1.0):
                raise ValueError("G must be a 0/1 adjacency matrix")
            if A[i][j] != A[j][i]:
                raise ValueError("G must be symmetric")
    names = [str(s) for s in ([statistics] if isinstance(statistics, str)
                              else statistics)]
    if not names:
        raise ValueError("at least one statistic is required")
    for nm in names:
        if nm not in _SUPPORTED:
            raise ValueError("unsupported statistic: %s" % nm)
    p = len(names)
    X = []
    yv = []
    for i in range(n):
        for j in range(i + 1, n):
            X.append(_change_stats(A, i, j, names))
            yv.append(A[i][j])
    nd = len(X)
    if nd < p:
        raise ValueError("fewer dyads than parameters")
    th = [0.0] * p if theta_init is None else [float(v) for v in theta_init]
    if len(th) != p:
        raise ValueError("theta_init must have one entry per statistic")
    used = 0
    for used in range(1, int(iters) + 1):
        H = [[0.0] * p for _ in range(p)]
        g = [0.0] * p
        for d in range(nd):
            eta = sum(X[d][k] * th[k] for k in range(p))
            mu = core.sigmoid(eta)
            wv = mu * (1.0 - mu)
            r = yv[d] - mu
            for a in range(p):
                g[a] += X[d][a] * r
                for c in range(p):
                    H[a][c] += wv * X[d][a] * X[d][c]
        try:
            step = core.cholsolve(H, g)
        except ValueError:
            raise ValueError(
                "pseudo-likelihood Hessian is singular: the dyad "
                "regression is separated or the change statistics are "
                "collinear on this graph")
        th = [th[k] + step[k] for k in range(p)]
        if max(abs(v) for v in step) < float(tol):
            break
    # observed information at the fit
    H = [[0.0] * p for _ in range(p)]
    ll = 0.0
    for d in range(nd):
        eta = sum(X[d][k] * th[k] for k in range(p))
        mu = core.sigmoid(eta)
        ll += yv[d] * math.log(mu) + (1.0 - yv[d]) * math.log(1.0 - mu)
        wv = mu * (1.0 - mu)
        for a in range(p):
            for c in range(p):
                H[a][c] += wv * X[d][a] * X[d][c]
    se = []
    try:
        for a in range(p):
            col = core.cholsolve(H, [1.0 if k == a else 0.0 for k in range(p)])
            se.append(math.sqrt(col[a]))
    except ValueError:
        raise ValueError(
            "pseudo-likelihood Hessian is singular: the dyad regression "
            "is separated or the change statistics are collinear on this "
            "graph")
    obs = []
    for nm in names:
        if nm == "edges":
            obs.append(sum(yv))
        elif nm == "twostar":
            deg = [sum(A[i]) for i in range(n)]
            obs.append(sum(d * (d - 1.0) / 2.0 for d in deg))
        else:
            t = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    if A[i][j]:
                        t += sum(1 for k in range(j + 1, n)
                                 if A[i][k] and A[j][k])
            obs.append(t)
    return RichResult(payload={
        "estimate": th[0],
        "theta": th,
        "se": se,
        "observed_stats": obs,
        "pseudo_loglik": ll,
        "n_dyads": nd,
        "iters_used": used,
        "n": n,
        "method": "Exponential random graph model (MPLE)",
    })


def cheatsheet():
    return "ergmod: Exponential random graph model"
