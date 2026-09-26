# morie.fn -- function file (rootcoder007/morie)
"""Generalized estimating equations (GEE)."""

from __future__ import annotations

import math

from . import _array_core as np
from . import _stats_core as _st
from ._containers import RegressionResult


def gee_regression(
    y: np.ndarray,
    X: np.ndarray,
    clusters: np.ndarray,
    *,
    family: str = "gaussian",
    corr_structure: str = "exchangeable",
    add_intercept: bool = True,
    max_iter: int = 50,
    tol: float = 1e-06,
) -> RegressionResult:
    """Generalized estimating equations with a robust (sandwich) covariance.

    Liang & Zeger (1986): solve :math:`\\sum_i D_i'V_i^{-1}(y_i - \\mu_i) = 0`
    with :math:`V_i = \\phi A_i^{1/2}R_i(\\alpha)A_i^{1/2}`, alternating a
    Fisher-scoring step in :math:`\\beta` with moment estimates of the scale
    and the working correlation from the Pearson residuals
    :math:`e_{ij} = (y_{ij} - \\mu_{ij})/\\sqrt{v(\\mu_{ij})}`:
    :math:`\\hat\\phi = \\sum e_{ij}^2/N`, and for ``"exchangeable"``
    :math:`\\hat\\alpha = \\sum_i\\sum_{j<k} e_{ij}e_{ik}/(\\hat\\phi\\sum_i n_i(n_i-1)/2)`,
    for ``"ar1"`` the least-squares fit of :math:`e_{ij}e_{ik}/\\hat\\phi` on
    :math:`\\alpha^{|j-k|}` over all within-cluster pairs (Prentice 1988),
    as geepack's geeglm computes them. Observations within a cluster are taken in their
    input order. The covariance is :math:`B^{-1}MB^{-1}` with
    :math:`B = \\sum D_i'V_i^{-1}D_i` and
    :math:`M = \\sum D_i'V_i^{-1}r_ir_i'V_i^{-1}D_i`.

    Parameters
    ----------
    y : array-like
        Response.
    X : array-like
        Covariates.
    clusters : array-like
        Cluster identifier of each row.
    family : str
        ``"gaussian"`` (identity link), ``"binomial"`` (logit) or
        ``"poisson"`` (log).
    corr_structure : str
        ``"independence"``, ``"exchangeable"`` or ``"ar1"``.
    add_intercept : bool
    max_iter : int
    tol : float

    Returns
    -------
    RegressionResult
        Coefficients, robust SEs and Wald p-values; ``extra`` holds
        ``alpha``, ``scale`` and the number of clusters.

    References
    ----------
    Liang, K.-Y. & Zeger, S. L. (1986). Longitudinal data analysis using
    generalized linear models. Biometrika 73, 13-22. Halekoh, U.,
    Hojsgaard, S. & Yan, J. (2006). The R package geepack for generalized
    estimating equations. Journal of Statistical Software 15(2).
    Prentice, R. L. (1988). Correlated binary regression with covariates
    specific to each binary observation. Biometrics 44, 1033-1048.
    """
    if family not in ("gaussian", "binomial", "poisson"):
        raise ValueError("family must be 'gaussian', 'binomial' or 'poisson'")
    if corr_structure not in ("independence", "exchangeable", "ar1"):
        raise ValueError("corr_structure must be 'independence', 'exchangeable' or 'ar1'")
    yv = [float(v) for v in np.asarray(y, dtype=float).ravel()]
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    n, p_raw = Xa.shape
    rows = [[float(v) for v in r] for r in Xa]
    if add_intercept:
        rows = [[1.0] + r for r in rows]
    k = len(rows[0])
    cl = list(np.asarray(clusters).ravel())
    groups = {}
    for i, c in enumerate(cl):
        groups.setdefault(c, []).append(i)
    order = list(groups.values())

    def inv_link(eta):
        if family == "binomial":
            return 1.0 / (1.0 + math.exp(-eta))
        if family == "poisson":
            return math.exp(eta)
        return eta

    def dmu(mu):
        if family == "binomial":
            return mu * (1.0 - mu)
        if family == "poisson":
            return mu
        return 1.0

    def var(mu):
        return dmu(mu) if family != "gaussian" else 1.0

    def solve(A, b):
        return [float(v) for v in np.linalg.solve(np.asarray(A), np.asarray(b))]

    # start from the independence GLM (IRLS), as geeglm does
    beta = [0.0] * k
    if family == "poisson":
        beta[0] = math.log(max(sum(yv) / n, 1e-8)) if add_intercept else 0.0
    for _ in range(100):
        mu = [inv_link(sum(r[j] * beta[j] for j in range(k))) for r in rows]
        w = [dmu(m) ** 2 / var(m) for m in mu]
        z = [sum(r[j] * beta[j] for j in range(k)) + (yv[i] - mu[i]) / dmu(mu[i]) for i, r in enumerate(rows)]
        A = [[sum(w[i] * rows[i][a] * rows[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
        bb = [sum(w[i] * rows[i][a] * z[i] for i in range(n)) for a in range(k)]
        new = solve(A, bb)
        done = max(abs(a - b) for a, b in zip(new, beta)) < 1e-12
        beta = new
        if done:
            break

    def moments(beta):
        mu = [inv_link(sum(r[j] * beta[j] for j in range(k))) for r in rows]
        e = [(yv[i] - mu[i]) / math.sqrt(var(mu[i])) for i in range(n)]
        phi = sum(v * v for v in e) / n
        alpha = 0.0
        if corr_structure == "exchangeable":
            s = sum(e[g[a]] * e[g[b]] for g in order for a in range(len(g)) for b in range(a + 1, len(g)))
            npair = sum(len(g) * (len(g) - 1) / 2 for g in order)
            alpha = s / (npair * phi) if npair else 0.0
        elif corr_structure == "ar1":
            pairs = [
                (e[g[a]] * e[g[b]] / phi, b - a) for g in order for a in range(len(g)) for b in range(a + 1, len(g))
            ]
            if pairs:
                # Prentice (1988): least squares of e_ij e_ik / phi on alpha^|j-k| over all pairs
                def obj(a):
                    return sum((zv - a**d) ** 2 for zv, d in pairs)

                lo, hi = -0.999999, 0.999999
                gr = (math.sqrt(5.0) - 1.0) / 2.0
                c, dd = hi - gr * (hi - lo), lo + gr * (hi - lo)
                fc, fd = obj(c), obj(dd)
                while hi - lo > 1e-15:
                    if fc < fd:
                        hi, dd, fd = dd, c, fc
                        c = hi - gr * (hi - lo)
                        fc = obj(c)
                    else:
                        lo, c, fc = c, dd, fd
                        dd = lo + gr * (hi - lo)
                        fd = obj(dd)
                alpha = 0.5 * (lo + hi)
        return mu, phi, alpha

    def pieces(beta, phi, alpha):
        mu = [inv_link(sum(r[j] * beta[j] for j in range(k))) for r in rows]
        B = np.zeros((k, k))
        U = np.zeros(k)
        M = np.zeros((k, k))
        for g in order:
            m = len(g)
            if corr_structure == "exchangeable":
                R = [[1.0 if a == b else alpha for b in range(m)] for a in range(m)]
            elif corr_structure == "ar1":
                R = [[alpha ** abs(a - b) for b in range(m)] for a in range(m)]
            else:
                R = [[1.0 if a == b else 0.0 for b in range(m)] for a in range(m)]
            sd = [math.sqrt(var(mu[i])) for i in g]
            V = np.asarray([[phi * sd[a] * R[a][b] * sd[b] for b in range(m)] for a in range(m)])
            D = np.asarray([[dmu(mu[i]) * rows[i][j] for j in range(k)] for i in g])
            r = np.asarray([yv[i] - mu[i] for i in g])
            VD = np.linalg.solve(V, D)
            Vr = np.linalg.solve(V, r)
            B = B + D.T @ VD
            U = U + D.T @ Vr
            s = D.T @ Vr
            M = M + np.outer(s, s)
        return B, U, M

    it = 0
    for step_no in range(1, int(max_iter) + 1):
        it = step_no
        _, phi, alpha = moments(beta)
        B, U, _ = pieces(beta, phi, alpha)
        step = [float(v) for v in np.linalg.solve(B, U)]
        beta = [a + b for a, b in zip(beta, step)]
        if max(abs(v) for v in step) < tol:
            break
    _, phi, alpha = moments(beta)
    B, _, M = pieces(beta, phi, alpha)
    Bi = np.linalg.inv(B)
    cov = Bi @ M @ Bi
    se_arr = [math.sqrt(max(float(cov[j, j]), 0.0)) for j in range(k)]
    z_vals = [b / s if s > 0 else float("nan") for b, s in zip(beta, se_arr)]
    p_vals = [float(2.0 * _st.norm.sf(abs(zv))) for zv in z_vals]
    mu_f = np.asarray([inv_link(sum(r[j] * beta[j] for j in range(k))) for r in rows])
    names = (["(Intercept)"] if add_intercept else []) + [f"x{j}" for j in range(p_raw)]
    return RegressionResult(
        method=f"GEE ({family}, {corr_structure})",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        fitted=mu_f,
        residuals=np.asarray(yv) - mu_f,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={
            "n_clusters": len(order),
            "corr_structure": corr_structure,
            "alpha": float(alpha),
            "scale": float(phi),
            "iterations": it,
        },
    )


gee = gee_regression


def cheatsheet() -> str:
    return "gee_regression({}) -> GEE with sandwich standard errors."


# compact alias per ledger/NAMING.md
geeregression = gee_regression
