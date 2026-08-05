# SPDX-License-Identifier: AGPL-3.0-or-later
"""Log-normal frailty for recurrent events."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["lnfrm", "lognormal_frailty"]


def _penalized_newton(t, e, Xa, D, sigma2, max_iter=50, tol=1e-10):
    # Newton on the penalized Breslow partial likelihood over the
    # augmented parameter (beta, b): l(beta, b) - b'b / (2 sigma2).
    n, p = Xa.shape
    q = D.shape[1]
    Z = np.concatenate([Xa, D], axis=1)
    theta = np.zeros(p + q)
    pen = np.zeros((p + q, p + q))
    for j in range(q):
        pen[p + j, p + j] = 1.0 / sigma2
    etimes = sorted(set(float(t[i]) for i in range(n) if e[i] == 1.0))
    info = None
    ll = 0.0
    for it in range(max_iter):
        eta = np.clip(Z @ theta, -500.0, 500.0)
        w = np.exp(eta)
        U = np.zeros(p + q)
        info = np.zeros((p + q, p + q))
        ll = 0.0
        for tk in etimes:
            Dk = [i for i in range(n) if t[i] == tk and e[i] == 1.0]
            R = [i for i in range(n) if t[i] >= tk]
            S0 = float(np.sum(np.asarray([w[i] for i in R])))
            S1 = np.zeros(p + q)
            S2 = np.zeros((p + q, p + q))
            for i in R:
                zi = Z[i]
                S1 = S1 + w[i] * zi
                S2 = S2 + w[i] * np.outer(zi, zi)
            d = float(len(Dk))
            xbar = S1 / S0
            for i in Dk:
                ll += float(eta[i])
                U = U + Z[i]
            ll -= d * float(np.log(S0))
            U = U - d * xbar
            info = info + d * (S2 / S0 - np.outer(xbar, xbar))
        b = theta[p:]
        ll_pen = ll - float(b @ b) / (2.0 * sigma2)
        U = U - pen @ theta
        info = info + pen
        step = np.linalg.solve(info, U)
        theta = theta + step
        if float(np.max(np.abs(step))) < tol:
            break
    return theta, info, ll_pen, it + 1, etimes


def lognormal_frailty(time, event, X, cluster, max_outer=50, tol=1e-7):
    """
    Log-normal frailty model by penalized partial likelihood with
    REML-type variance update (McGilchrist 1993).

    Hazard lambda_ij(t | b_i) = lambda_0(t) exp(beta' x_ij + b_i) with
    b_i ~ N(0, sigma^2) per cluster. For fixed sigma^2 the BLUP-type
    estimates maximise the penalized Breslow partial likelihood
    l(beta, b) - b'b / (2 sigma^2); sigma^2 is then updated by the REML
    formula sigma^2 = (b'b + trace of the frailty block of the inverse
    penalized information) / q, and the two steps alternate.

    References: McGilchrist, C. A. (1993), "REML estimation for
    survival models with frailty", Biometrics 49(1), 221-225;
    McGilchrist & Aisbett (1991), Biometrics 47(2), 461-466;
    Therneau, Grambsch & Pankratz (2003), J Comp Graph Stat 12(1),
    156-175 (penalized formulation).

    Returns
    -------
    result : RichResult
        Keys: estimate (beta), se, frailty (cluster -> b), sigma2,
        loglik_penalized, n_outer, n_newton.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=float)
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape((-1, 1))
    if Xa.shape[0] != t.shape[0]:
        Xa = Xa.T
    n, p = Xa.shape
    cl = list(cluster)
    ks = sorted(set(cl))
    q = len(ks)
    if q < 2:
        raise ValueError("need at least 2 clusters")
    D = np.zeros((n, q))
    for i in range(n):
        D[i, ks.index(cl[i])] = 1.0
    sigma2 = 0.5
    theta = None
    for outer in range(max_outer):
        theta, info, ll_pen, n_newton, etimes = _penalized_newton(
            t, e, Xa, D, sigma2)
        b = theta[p:]
        cov = np.linalg.inv(info)
        tr = sum(float(cov[p + j, p + j]) for j in range(q))
        sigma2_new = (float(b @ b) + tr) / q
        if abs(sigma2_new - sigma2) < tol * max(1.0, sigma2):
            sigma2 = sigma2_new
            break
        sigma2 = sigma2_new
    theta, info, ll_pen, n_newton, etimes = _penalized_newton(t, e, Xa, D, sigma2)
    cov = np.linalg.inv(info)
    dg = [float(cov[j, j]) for j in range(p)]
    if any(v <= 0.0 or v != v for v in dg):
        raise ValueError("penalized information is singular")
    b = theta[p:]
    return RichResult(payload={
        "estimate": theta[:p],
        "se": np.sqrt(np.asarray(dg)),
        "frailty": {ks[j]: float(b[j]) for j in range(q)},
        "sigma2": sigma2,
        "loglik_penalized": ll_pen,
        "n_outer": outer + 1,
        "n_newton": n_newton,
        "method": "McGilchrist (1993) log-normal frailty, penalized PL + REML variance",
    })


lnfrm = lognormal_frailty


def cheatsheet():
    return "lnfrm(time, event, X, cluster) -> log-normal frailty Cox via penalized partial likelihood."
