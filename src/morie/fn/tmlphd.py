# morie.fn -- function file (rootcoder007/morie)
"""High-dimensional TMLE with L1-penalised nuisance models."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_high_dim"]


def _soft(z, t):
    if z > t:
        return z - t
    if z < -t:
        return z + t
    return 0.0


def _lasso(X, y, lam, w=None, sweeps=400):
    """Cyclic coordinate descent for a weighted lasso; column 0 unpenalised.

    Fixed sweep count and a fixed cyclic order, so both language arms
    take exactly the same path even where the objective has a flat
    direction.
    """
    n = len(y)
    p = len(X[0])
    if w is None:
        w = [1.0] * n
    beta = [0.0] * p
    fit = [0.0] * n
    denom = [sum(w[i] * X[i][j] * X[i][j] for i in range(n)) / n for j in range(p)]
    for _ in range(sweeps):
        for j in range(p):
            if denom[j] <= 0.0:
                continue
            r = sum(w[i] * X[i][j] * (y[i] - fit[i] + X[i][j] * beta[j]) for i in range(n)) / n
            nb = r / denom[j] if j == 0 else _soft(r, lam) / denom[j]
            if nb != beta[j]:
                dlt = nb - beta[j]
                for i in range(n):
                    fit[i] += dlt * X[i][j]
                beta[j] = nb
    return beta


def _lasso_logit(X, y, lam, outer=15, sweeps=60):
    """Proximal-Newton L1 logistic: IRLS outside, weighted lasso inside."""
    n = len(y)
    p = len(X[0])
    beta = [0.0] * p
    for _ in range(outer):
        eta = [C.dot(X[i], beta) for i in range(n)]
        mu = [S.expit(e) for e in eta]
        w = [S.clip(m * (1.0 - m), 1e-6, 0.25) for m in mu]
        z = [eta[i] + (y[i] - mu[i]) / w[i] for i in range(n)]
        beta = _lasso(X, z, lam, w=w, sweeps=sweeps)
    return beta


def tmle_high_dim(y, D, X, lam):
    """Targeted ATE with lasso-fitted outcome and propensity models.

    When ``p`` is comparable with ``n`` the two nuisance models must be
    regularised, and regularisation is exactly what breaks a plain
    plug-in: the penalty biases ``Q`` towards zero and that bias does not
    vanish at root-n.  The targeting step removes it, which is why the
    post-selection literature and the TMLE literature arrive at the same
    correction from different directions.  Selection is by L1 on both
    models -- coordinate descent for ``Q``, proximal-Newton for ``g`` --
    with the intercept left unpenalised, followed by the usual
    fluctuation ``eps = sum H (y - Q)/sum H^2``,
    ``H = D/g - (1 - D)/(1 - g)``.

    Determinism: a fixed number of coordinate-descent sweeps in a fixed
    cyclic order, no active-set heuristics and no convergence tolerance.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Covariates; ``p`` may exceed ``n``.
    lam : float
        L1 penalty, applied to both nuisance models.  Must be >= 0.
        ``lam = 0`` reproduces the unpenalised TMLE.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``nz_q``, ``nz_g``, ``n``.

    References
    ----------
    Belloni, A., Chernozhukov, V. & Hansen, C. (2014).  Inference on
    treatment effects after selection among high-dimensional controls.
    Review of Economic Studies 81(2):608-650.  doi:10.1093/restud/rdt044.
    The targeting step is van der Laan, M. J. & Rubin, D. (2006), IJB
    2(1):11.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    n = len(yv)
    lam = float(lam)
    if n == 0 or len(Dv) != n:
        raise ValueError("tmle_high_dim: y and D must share one length")
    if lam < 0.0:
        raise ValueError("tmle_high_dim: lam must be non-negative")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_high_dim: X must have one row per subject")
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    gb = _lasso_logit(W, Dv, lam)
    g = [S.clip(S.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]
    des = [[1.0, Dv[i]] + list(Xm[i]) for i in range(n)]
    qb = _lasso(des, yv, lam)

    def qhat(i, a):
        return C.dot([1.0, a] + list(Xm[i]), qb)

    Q1 = [qhat(i, 1.0) for i in range(n)]
    Q0 = [qhat(i, 0.0) for i in range(n)]
    Qobs = [qhat(i, Dv[i]) for i in range(n)]
    H = [Dv[i] / g[i] - (1.0 - Dv[i]) / (1.0 - g[i]) for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (yv[i] - Qobs[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Q1s = [Q1[i] + eps / g[i] for i in range(n)]
    Q0s = [Q0[i] - eps / (1.0 - g[i]) for i in range(n)]
    psi = sum(Q1s[i] - Q0s[i] for i in range(n)) / n
    ic = [H[i] * (yv[i] - Qobs[i] - eps * H[i]) + Q1s[i] - Q0s[i] - psi for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    nzq = float(sum(1 for j in range(2, len(qb)) if qb[j] != 0.0))
    nzg = float(sum(1 for j in range(1, len(gb)) if gb[j] != 0.0))
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps, "nz_q": nzq, "nz_g": nzg, "n": n,
        "method": "High-dimensional TMLE with L1-penalised nuisance models"})


def cheatsheet():
    return "tmlphd: high-dimensional TMLE with lasso nuisance models."
