# morie.fn -- function file (rootcoder007/morie)
"""Spatial probit ML with GHK simulator."""

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import random_uniform
from ._rrng_core import qnorm
from .spdiscrete import binary_glm
from .spprmf import _rows


def _logphi(x):
    """log Phi(x), accurate in the lower tail."""
    if x > -5.0:
        return math.log(0.5 * math.erfc(-x / math.sqrt(2.0)))
    return math.log(0.5 * math.erfc(-x / math.sqrt(2.0)) + 1e-300)


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for j in range(n):
        s = A[j][j] - ssum(L[j][k] * L[j][k] for k in range(j))
        L[j][j] = math.sqrt(s)
        for i in range(j + 1, n):
            L[i][j] = (A[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]
    return L


def _ghk_loglik(par, yv, Xm, Wm, U):
    """Simulated log-likelihood of the SAR probit, rho = tanh(theta) (common random numbers U)."""
    n, p = len(yv), len(Xm[0])
    b, rho = par[:p], math.tanh(par[p])
    Si = inverse([[(1.0 if i == j else 0.0) - rho * Wm[i][j] for j in range(n)] for i in range(n)])
    xb = [ssum(r[k] * b[k] for k in range(p)) for r in Xm]
    s = [2.0 * v - 1.0 for v in yv]
    m = [s[i] * ssum(Si[i][j] * xb[j] for j in range(n)) for i in range(n)]
    C = [[s[i] * s[j] * ssum(Si[i][k] * Si[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    L = _chol(C)
    logs = []
    for u in U:
        eta = [0.0] * n
        lp = 0.0
        for i in range(n):
            c = (m[i] + ssum(L[i][j] * eta[j] for j in range(i))) / L[i][i]
            lq = _logphi(c)
            lp += lq
            q = (1.0 - u[i]) * math.exp(lq)
            eta[i] = -qnorm(min(max(q, 1e-300), 1.0 - 1e-16))
        logs.append(lp)
    top = max(logs)
    return top + math.log(ssum(math.exp(v - top) for v in logs) / len(logs))


def _nelder_mead(f, x0, xtol=1e-10, ftol=1e-12, maxiter=20000):
    """Nelder-Mead minimisation (Lagarias et al. 1998 coefficients, fminsearch initial simplex)."""
    k = len(x0)
    pts = [list(x0)]
    for j in range(k):
        x = list(x0)
        x[j] = x[j] * 1.05 if x[j] != 0.0 else 0.00025
        pts.append(x)
    vals = [f(x) for x in pts]
    it = 0
    while it < maxiter:
        order = sorted(range(k + 1), key=lambda i: vals[i])
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if (
            max(abs(v - vals[0]) for v in vals[1:]) <= ftol
            and max(abs(pts[i][j] - pts[0][j]) for i in range(1, k + 1) for j in range(k)) <= xtol
        ):
            break
        it += 1
        cen = [ssum(pts[i][j] for i in range(k)) / k for j in range(k)]
        xr = [2.0 * cen[j] - pts[k][j] for j in range(k)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [3.0 * cen[j] - 2.0 * pts[k][j] for j in range(k)]
            fe = f(xe)
            if fe < fr:
                pts[k], vals[k] = xe, fe
            else:
                pts[k], vals[k] = xr, fr
        elif fr < vals[k - 1]:
            pts[k], vals[k] = xr, fr
        else:
            if fr < vals[k]:
                xc = [1.5 * cen[j] - 0.5 * pts[k][j] for j in range(k)]
                fc = f(xc)
                accept = fc <= fr
            else:
                xc = [0.5 * cen[j] + 0.5 * pts[k][j] for j in range(k)]
                fc = f(xc)
                accept = fc < vals[k]
            if accept:
                pts[k], vals[k] = xc, fc
            else:
                for i in range(1, k + 1):
                    pts[i] = [pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(k)]
                    vals[i] = f(pts[i])
    best = min(range(k + 1), key=lambda i: vals[i])
    return pts[best], vals[best], it


def spprml(y, X, W, nsim=9, seed=0, xtol=1e-10, maxiter=20000):
    r"""SAR probit by simulated maximum likelihood with the GHK simulator (Beron and Vijverberg 2004).

    y* = rho W y* + X beta + e, e ~ N(0, I), y = 1(y* > 0). With
    S = I - rho W the latent vector is N(S^{-1} X beta, (S'S)^{-1}) and
    the likelihood is the orthant probability of z = D y* (D =
    diag(2y - 1)), evaluated by the Geweke-Hajivassiliou-Keane recursive
    simulator on the Cholesky factor of D (S'S)^{-1} D with nsim
    common Philox uniform vectors (stream r of seed), so the
    simulated log-likelihood is smooth in the parameters. It is maximised by
    Nelder-Mead over (beta, atanh(rho)) from the probit estimates and
    rho = 0. An intercept is prepended when X has no constant column.
    Use a larger nsim in practice; the simulation bias is O(1/nsim).

    References
    ----------
    Beron, K. J. and Vijverberg, W. P. M. (2004). Probit in a spatial context:
    a Monte Carlo analysis. In L. Anselin, R. J. G. M. Florax and S. J. Rey
    (eds), *Advances in Spatial Econometrics*. Springer, 169-195.
    Train, K. E. (2009). *Discrete Choice Methods with Simulation*, 2nd ed.,
    sec. 5.6.3. Cambridge University Press.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
    >>> r = spprml([1, 0, 1, 1, 0, 0, 1, 0, 0, 1], X, W, nsim=20)
    >>> round(r["rho"], 6)
    -0.574097
    """
    yv = [float(v) for v in y]
    Xm = _rows(X)
    if not any(len({r[c] for r in Xm}) == 1 and Xm[0][c] != 0.0 for c in range(len(Xm[0]))):
        Xm = [[1.0] + r for r in Xm]
    Wm = _rows(W)
    n, p = len(yv), len(Xm[0])
    U = [[float(v) for v in random_uniform(n, seed=seed, stream=r)] for r in range(int(nsim))]
    start = list(binary_glm(yv, Xm, link="probit")["coefficients"]) + [0.0]
    x, fval, it = _nelder_mead(lambda q: -_ghk_loglik(q, yv, Xm, Wm, U), start, xtol=xtol, maxiter=maxiter)
    return RichResult(
        payload={"coefficients": x[:p], "rho": math.tanh(x[p]), "loglik": -fval, "iterations": it, "nsim": int(nsim)}
    )


spprml_fn = spprml


def cheatsheet() -> str:
    return "spprml(y, X, W, nsim=9) -> SAR probit by GHK simulated ML (Beron and Vijverberg 2004)."
