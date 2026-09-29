# morie.fn -- function file (rootcoder007/morie)
"""GWR Poisson regression."""

import math

from ._qpcore import ssum
from ._richresult import RichResult
from .gwrcoef import _setup, _wls


def _ggwr(y, X, coords, bw, kernel, adaptive, tol, maxiter, family):
    yv, Xm, _, _, Wc = _setup(y, X, coords, bw, kernel, adaptive)
    n, p = len(yv), len(Xm[0])
    if family == "poisson":
        mu = [v + 0.1 for v in yv]
        nu = [math.log(v) for v in mu]
    else:
        mu = [0.5] * n
        nu = [0.0] * n
    wt2 = [1.0] * n
    llik = 0.0
    it = 0
    while True:
        if family == "poisson":
            yadj = [nu[j] + (yv[j] - mu[j]) / mu[j] for j in range(n)]
        else:
            yadj = [nu[j] + (yv[j] - mu[j]) / (mu[j] * (1.0 - mu[j])) for j in range(n)]
        fits = [_wls(Xm, [Wc[i][j] * wt2[j] for j in range(n)], yadj) for i in range(n)]
        betas = [f[0] for f in fits]
        nu = [ssum(Xm[i][a] * betas[i][a] for a in range(p)) for i in range(n)]
        if family == "poisson":
            mu = [math.exp(v) for v in nu]
            new = ssum(yv[j] * math.log(mu[j]) - mu[j] - math.lgamma(yv[j] + 1.0) for j in range(n))
        else:
            mu = [1.0 / (1.0 + math.exp(-v)) for v in nu]
            new = ssum(yv[j] * math.log(mu[j]) + (1.0 - yv[j]) * math.log(1.0 - mu[j]) for j in range(n))
        old, llik = llik, new
        if abs((old - llik) / llik) < tol:
            break
        wt2 = list(mu) if family == "poisson" else [m * (1.0 - m) for m in mu]
        it += 1
        if it == maxiter:
            break
    se = [[math.sqrt(ssum(C[a][j] * C[a][j] / wt2[j] for j in range(n))) for a in range(p)] for _, C in fits]
    if family == "poisson":
        dev = ssum(
            2.0 * (yv[j] * (math.log(yv[j] / mu[j]) - 1.0) + mu[j]) if yv[j] != 0 else 2.0 * mu[j] for j in range(n)
        )
    else:
        dev = -2.0 * llik
    return RichResult(
        payload={"betas": betas, "se": se, "fitted": mu, "loglik": llik, "deviance": dev, "iterations": it}
    )


def gwrpois(y, X, coords, bw=0.5, kernel="bisquare", adaptive=False, tol=1e-10, maxiter=200):
    r"""Geographically weighted Poisson regression (Nakaya et al. 2005) by the GWmodel local scoring.

    Iteratively reweighted least squares over all locations at once: with the
    current linear predictors ``eta_j`` and means ``mu_j = exp(eta_j)`` the
    working response ``z_j = eta_j + (y_j - mu_j)/mu_j`` is regressed at each
    location ``i`` by WLS with weights ``w_ij mu_j`` (``mu_j = 1`` at the
    first pass, starting from ``mu = y + 0.1``); ``eta_i = x_i beta_i`` is
    updated and the loop stops when the relative change of the Poisson
    log-likelihood is below ``tol`` or after ``maxiter`` passes. With
    ``tol=1e-5, maxiter=20`` this is ``GWmodel::ggwr.basic(family =
    "poisson")`` exactly; the default iterates to convergence. Standard
    errors are ``sqrt(diag(C_i diag(1/mu) C_i'))`` and the deviance is the
    Poisson deviance.

    References
    ----------
    Nakaya, T., Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2005).
    Geographically weighted Poisson regression for disease association
    mapping. *Statistics in Medicine* 24, 2695-2717.

    Examples
    --------
    >>> P = [(float(i % 5), float(i // 5)) for i in range(20)]
    >>> X = [[(0.37 * i) % 1.3] for i in range(20)]
    >>> y = [float((i * 7) % 5 + (i // 5)) for i in range(20)]
    >>> r = gwrpois(y, X, P, 4.0, kernel="gaussian")
    >>> round(r["betas"][0][1], 8)
    0.47575566
    """
    return _ggwr(y, X, coords, bw, kernel, adaptive, tol, maxiter, "poisson")


gwrpois_fn = gwrpois


def cheatsheet() -> str:
    return "gwrpois(y, X, coords, bw) -> GW Poisson regression by local scoring (GWmodel::ggwr.basic)."
