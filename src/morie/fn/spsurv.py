# morie.fn -- function file (rootcoder007/morie)
"""Spatial survival and SPDE fields: the Weibull proportional-hazards model with censoring and a
shared (area-level) gamma frailty by maximum marginal likelihood, and the SPDE (Matern, alpha = 2)
Gaussian Markov random field precision on a regular grid."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from .lbfgsb import lbfgsb_minimize

__all__ = ["weibull_frailty_fit", "spde_precision_grid"]


def _digamma(x):
    r = 0.0
    while x < 10.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f / 132))))


def weibull_frailty_fit(time, event, X, cluster=None, *, max_iter: int = 500) -> RichResult:
    r"""Weibull proportional hazards with right censoring and an optional shared gamma frailty.

    Hazard ``h(t) = lambda rho t^(rho - 1) exp(x' beta)``; the likelihood of
    each subject is ``f(t)^delta S(t)^(1 - delta) = h(t)^delta S(t)``
    (Lawson 2021, eq. 15.1). With ``cluster`` labels a gamma frailty of mean
    1 and variance ``theta`` is shared within clusters and integrated out:
    ``prod_i h_i^delta_i Gamma(1/theta + D) / Gamma(1/theta) theta^D (1 + theta H)^(-(1/theta + D))``
    with ``D`` the events and ``H`` the cumulative hazard of the cluster.
    Maximised over ``(log lambda, log rho, beta, log theta)`` by L-BFGS with
    the analytic score and Newton polishing; standard errors from the inverse
    numerical Hessian of the score.

    References
    ----------
    Lawson, A. B. (2021). Using R for Bayesian Spatial and Spatio-Temporal
    Health Modeling, ch. 15. Klein, J. P. (1992). Semiparametric estimation
    of random effects using the Cox model based on the EM algorithm.
    Biometrics 48, 795-806. Duchateau, L. and Janssen, P. (2008). The Frailty Model. Springer.

    Examples
    --------
    >>> t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> r = weibull_frailty_fit(t, [1, 1, 1, 1, 1, 1], [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    >>> round(r.rho, 6) > 0, len(r.beta)
    (True, 1)
    """
    ts = [float(v) for v in time]
    dl = [float(v) for v in event]
    Xm = [[float(v) for v in r] for r in X]
    n, p = len(ts), len(Xm[0])
    lt = [math.log(v) for v in ts]
    groups = None
    if cluster is not None:
        labels = sorted(set(cluster), key=lambda v: (str(type(v)), v))
        groups = [[i for i in range(n) if cluster[i] == g] for g in labels]

    def fg(par):
        la, lr = par[0], par[1]
        beta = par[2 : 2 + p]
        lam, rho = math.exp(la), math.exp(lr)
        eta = [ssum(Xm[i][k] * beta[k] for k in range(p)) for i in range(n)]
        Hi = [lam * ts[i] ** rho * math.exp(eta[i]) for i in range(n)]
        f = ssum(dl[i] * (la + lr + (rho - 1) * lt[i] + eta[i]) for i in range(n))
        g = [0.0] * len(par)
        g[0] = ssum(dl)
        g[1] = ssum(dl[i] * (1 + rho * lt[i]) for i in range(n))
        for k in range(p):
            g[2 + k] = ssum(dl[i] * Xm[i][k] for i in range(n))
        if groups is None:
            f -= ssum(Hi)
            g[0] -= ssum(Hi)
            g[1] -= ssum(Hi[i] * rho * lt[i] for i in range(n))
            for k in range(p):
                g[2 + k] -= ssum(Hi[i] * Xm[i][k] for i in range(n))
            return -f, [-v for v in g]
        th = math.exp(par[2 + p])
        a = 1.0 / th
        gth = 0.0
        for idx in groups:
            D = ssum(dl[i] for i in idx)
            H = ssum(Hi[i] for i in idx)
            f += math.lgamma(a + D) - math.lgamma(a) + D * math.log(th) - (a + D) * math.log(1 + th * H)
            c = (a + D) * th / (1 + th * H)
            g[0] -= c * H
            g[1] -= c * ssum(Hi[i] * rho * lt[i] for i in idx)
            for k in range(p):
                g[2 + k] -= c * ssum(Hi[i] * Xm[i][k] for i in idx)
            gth += -a * (_digamma(a + D) - _digamma(a)) + D + a * math.log(1 + th * H) - (a + D) * th * H / (1 + th * H)
        g[2 + p] = gth
        return -f, [-v for v in g]

    k = 2 + p + (0 if groups is None else 1)
    x0 = [math.log(ssum(dl) / ssum(ts)), 0.0] + [0.0] * p + ([math.log(0.5)] if groups is not None else [])
    res = lbfgsb_minimize(
        lambda v: fg(list(v))[0], x0, grad=lambda v: fg(list(v))[1], pgtol=1e-10, factr=10.0, max_iter=max_iter
    )
    x = [float(v) for v in res.x]

    def hess(x):
        H = []
        for a in range(k):
            h = 1e-5 * max(1.0, abs(x[a]))
            up, dn = list(x), list(x)
            up[a] += h
            dn[a] -= h
            gu, gd = fg(up)[1], fg(dn)[1]
            H.append([(u - v) / (2 * h) for u, v in zip(gu, gd)])
        return [[(H[a][b] + H[b][a]) / 2 for b in range(k)] for a in range(k)]

    # Newton polish on the analytic score
    for _ in range(20):
        g = fg(x)[1]
        H = hess(x)
        step = [ssum(v * w for v, w in zip(row, g)) for row in inverse(H)]
        new = [a - b for a, b in zip(x, step)]
        if fg(new)[0] > fg(x)[0] + 1e-12 * abs(fg(x)[0]):
            break
        done = max(abs(v) / max(1.0, abs(a)) for v, a in zip(step, x)) <= 1e-13
        x = new
        if done:
            break
    H = hess(x)
    V = inverse(H)
    out = {
        "lambda_": math.exp(x[0]),
        "rho": math.exp(x[1]),
        "beta": x[2 : 2 + p],
        "loglik": -fg(x)[0],
        "se_log_params": [math.sqrt(max(V[a][a], 0.0)) for a in range(k)],
    }
    if groups is not None:
        out["theta"] = math.exp(x[2 + p])
    return RichResult(payload=out)


def spde_precision_grid(nx: int, ny: int, kappa: float, tau: float, *, h: float = 1.0) -> RichResult:
    r"""SPDE (Matern, alpha = 2) GMRF precision on a regular ``nx x ny`` grid (Lindgren, Rue and Lindstrom 2011).

    ``Q = tau^2 (kappa^4 C + 2 kappa^2 G + G C^(-1) G)`` with lumped mass
    ``C = h^2 I`` and stiffness ``G`` the Neumann 5-point graph Laplacian
    (degree on the diagonal, -1 for grid neighbours). The field
    approximates a Matern field with ``nu = 1``, range ``sqrt(8) / kappa``
    and marginal variance ``1 / (4 pi kappa^2 tau^2)``; node ``i + nx j`` is
    column ``i``, row ``j``.

    References
    ----------
    Lindgren, F., Rue, H. and Lindstrom, J. (2011). An explicit link between
    Gaussian fields and Gaussian Markov random fields: the stochastic partial
    differential equation approach. JRSS B 73, 423-498.

    Examples
    --------
    >>> r = spde_precision_grid(2, 1, 1.0, 1.0)
    >>> r.Q
    [[5.0, -4.0], [-4.0, 5.0]]
    """
    n = nx * ny
    G = [[0.0] * n for _ in range(n)]
    for j in range(ny):
        for i in range(nx):
            a = i + nx * j
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    G[a][ii + nx * jj] = -1.0
                    G[a][a] += 1.0
    c = h * h
    GG = [[ssum(G[a][q] * G[q][b] for q in range(n)) / c for b in range(n)] for a in range(n)]
    Q = [
        [tau * tau * ((kappa**4 * c if a == b else 0.0) + 2 * kappa**2 * G[a][b] + GG[a][b]) for b in range(n)]
        for a in range(n)
    ]
    return RichResult(
        payload={"Q": Q, "range": math.sqrt(8.0) / kappa, "marginal_variance": 1.0 / (4 * math.pi * kappa**2 * tau**2)}
    )


def cheatsheet() -> str:
    return "weibull_frailty_fit / spde_precision_grid -> spatial survival (gamma frailty) and SPDE precision."
