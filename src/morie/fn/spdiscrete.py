# morie.fn -- function file (rootcoder007/morie)
"""Spatial binary-choice models estimated without latent-variable integration: the linearized GMM
spatial logit of Klier and McMillen (2008), the GMM spatial autoregressive probit of Pinkse and
Slade (1998) as implemented by McMillen, and the Kelejian-Prucha Moran test for spatial
dependence in probit and logit generalized residuals."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult
from ._rrng_core import dnorm, pnorm

__all__ = ["binary_glm", "spatial_logit_gmm", "spatial_probit_gmm", "discrete_moran_test"]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _mat(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in r] for r in a.tolist()]


def _mv(A, v):
    return [ssum(r[j] * v[j] for j in range(len(v))) for r in A]


def _cols(M):
    return [list(c) for c in zip(*M)]


def _ls(G, u):
    k = len(G[0])
    return solve(
        [[ssum(r[a] * r[b] for r in G) for b in range(k)] for a in range(k)],
        [ssum(r[a] * v for r, v in zip(G, u)) for a in range(k)],
    )


def _project(Z, v):
    b = _ls(Z, v)
    return [ssum(r[a] * b[a] for a in range(len(b))) for r in Z]


def _cdf(e, link):
    return 1.0 / (1.0 + math.exp(-e)) if link == "logit" else pnorm(e)


def _pdf(e, link):
    if link == "logit":
        p = 1.0 / (1.0 + math.exp(-e))
        return p * (1 - p)
    return dnorm(e)


def binary_glm(y, X, link="logit", tol=1e-12, maxit=100):
    r"""Logit or probit maximum likelihood by iteratively reweighted least squares.

    ``X`` should include the intercept column. Stops when the relative change
    in the deviance is below ``tol``.

    Examples
    --------
    >>> r = binary_glm([1, 0, 1, 1, 0, 0], [[1, 2.0], [1, -1.0], [1, 0.1], [1, 1.5], [1, 0.6], [1, -0.4]])
    >>> [round(b, 6) for b in r.coefficients]
    [-0.913755, 2.181707]
    """
    yv, Xm = _vec(y), _mat(X)
    k = len(Xm[0])
    beta = [0.0] * k
    dev_old = math.inf
    for _ in range(maxit):
        eta = _mv(Xm, beta)
        mu = [min(max(_cdf(e, link), 1e-15), 1 - 1e-15) for e in eta]
        d = [_pdf(e, link) for e in eta]
        w = [di * di / (m * (1 - m)) for di, m in zip(d, mu)]
        z = [e + (yy - m) / di for e, yy, m, di in zip(eta, yv, mu, d)]
        A = [[ssum(w[i] * Xm[i][a] * Xm[i][b] for i in range(len(yv))) for b in range(k)] for a in range(k)]
        beta = solve(A, [ssum(w[i] * Xm[i][a] * z[i] for i in range(len(yv))) for a in range(k)])
        eta = _mv(Xm, beta)
        mu = [min(max(_cdf(e, link), 1e-15), 1 - 1e-15) for e in eta]
        dev = -2 * ssum(yy * math.log(m) + (1 - yy) * math.log(1 - m) for yy, m in zip(yv, mu))
        if abs(dev - dev_old) / (abs(dev) + 0.1) < tol:
            break
        dev_old = dev
    return RichResult(payload={"coefficients": beta, "fitted": mu, "linear_predictor": eta, "deviance": dev})


def spatial_logit_gmm(y, X, W, Z=None):
    r"""Klier and McMillen (2008) linearized GMM spatial autoregressive logit.

    Starting from the standard logit (``p``, ``xb``), the model is linearized
    around ``rho = 0``: the pseudo-response ``u = y - p + g X b`` (``g = p(1 -
    p)``) is regressed on the instrumented gradients ``[g X, g W xb]``, each
    replaced by its projection on the instruments ``Z`` (default ``[X, W
    X]`` without the lagged intercept). The last coefficient is ``rho``;
    standard errors are HC3 (``car::hccm``), as in ``McSpatial::splogit``.

    References
    ----------
    Klier, T. and McMillen, D. P. (2008). Clustering of auto supplier plants
    in the United States: generalized method of moments spatial logit for
    large samples. *Journal of Business and Economic Statistics* 26, 460-471.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
    >>> y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
    >>> r = spatial_logit_gmm(y, X, W)
    >>> round(r.rho, 10)
    -0.5078680767
    """
    yv, Xm, Wm = _vec(y), _mat(X), _mat(W)
    n, k = len(Xm), len(Xm[0])
    fit = binary_glm(yv, Xm, "logit")
    b, xb, p = fit.coefficients, fit.linear_predictor, fit.fitted
    g = [q * (1 - q) for q in p]
    if Z is None:
        wx = [_mv(Wm, c) for c in _cols(Xm)[1:]]
        Zm = [r + [c[i] for c in wx] for i, r in enumerate(Xm)]
    else:
        Zm = _mat(Z)
    wxb = _mv(Wm, xb)
    cols = [[g[i] * Xm[i][a] for i in range(n)] for a in range(k)] + [[g[i] * wxb[i] for i in range(n)]]
    u = [yv[i] - p[i] + ssum(cols[a][i] * b[a] for a in range(k)) for i in range(n)]
    G = _cols([_project(Zm, c) for c in cols])
    coef = _ls(G, u)
    se = _hc3(G, u, coef)
    return RichResult(payload={"coefficients": coef[:k], "rho": coef[k], "se": se, "logit": b})


def _hc3(G, u, coef):
    n, k = len(G), len(G[0])
    B = inverse([[ssum(r[a] * r[b] for r in G) for b in range(k)] for a in range(k)])
    e = [v - ssum(r[a] * coef[a] for a in range(k)) for r, v in zip(G, u)]
    h = [ssum(G[i][a] * ssum(B[a][b] * G[i][b] for b in range(k)) for a in range(k)) for i in range(n)]
    w = [ei * ei / (1 - hi) ** 2 for ei, hi in zip(e, h)]
    M = [[ssum(w[i] * G[i][a] * G[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    V = [
        [ssum(B[a][c] * ssum(M[c][d] * B[d][b] for d in range(k)) for c in range(k)) for b in range(k)]
        for a in range(k)
    ]
    return [math.sqrt(V[a][a]) for a in range(k)]


def spatial_probit_gmm(y, X, W, Z=None, start_rho=0.0, tol=1e-10, maxit=500):
    r"""GMM spatial autoregressive probit (Pinkse and Slade 1998), McMillen's iteration.

    With ``A = (I - rho W)^{-1}`` and ``s_i = sqrt((A A')_ii)``, the index is
    ``x*_i = (A X beta)_i / s_i`` and the generalized residuals ``u = (y - Phi)
    phi / (Phi (1 - Phi))``. Gauss-Newton steps regress ``u`` on the gradient
    ``d u / d(beta, rho)`` (``du = u (x* + u)`` times ``A X / s`` and ``A W
    x* - diag(A A' (W + W' - 2 rho W'W) A A') x* / (2 s^2)``), each column
    projected on the instruments ``Z`` (default ``[X, W X]``), until the
    largest step is below ``tol``. Standard errors use the sandwich ``(G'G)^{-1}
    G' diag(u^2) G (G'G)^{-1}``. ``tol = 1e-4`` reproduces
    ``McSpatial::gmmprobit``.

    References
    ----------
    Pinkse, J. and Slade, M. E. (1998). Contracting in space: an application of
    spatial statistics to discrete-choice models. *Journal of Econometrics* 85,
    125-154.

    Klier, T. and McMillen, D. P. (2008). *Journal of Business and Economic
    Statistics* 26, 460-471.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
    >>> y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
    >>> r = spatial_probit_gmm(y, X, W)
    >>> round(r.rho, 8), r.iterations
    (-0.5403822, 14)
    """
    yv, Xm, Wm = _vec(y), _mat(X), _mat(W)
    n, k = len(Xm), len(Xm[0])
    if Z is None:
        wx = [_mv(Wm, c) for c in _cols(Xm)[1:]]
        Zm = [r + [c[i] for c in wx] for i, r in enumerate(Xm)]
    else:
        Zm = _mat(Z)
    b = binary_glm(yv, Xm, "probit").coefficients + [float(start_rho)]
    WWt = [[Wm[i][j] + Wm[j][i] for j in range(n)] for i in range(n)]
    WtW = [[ssum(Wm[m][i] * Wm[m][j] for m in range(n)) for j in range(n)] for i in range(n)]

    def step(bb):
        beta, rho = bb[:k], bb[k]
        Ai = inverse([[(1.0 if i == j else 0.0) - rho * Wm[i][j] for j in range(n)] for i in range(n)])
        V = [[ssum(Ai[i][m] * Ai[j][m] for m in range(n)) for j in range(n)] for i in range(n)]
        sv = [math.sqrt(V[i][i]) for i in range(n)]
        AX = [[ssum(Ai[i][m] * Xm[m][a] for m in range(n)) / sv[i] for a in range(k)] for i in range(n)]
        xs = [ssum(AX[i][a] * beta[a] for a in range(k)) for i in range(n)]
        cp = [pnorm(v) for v in xs]
        dp = [dnorm(v) for v in xs]
        u = [(yv[i] - cp[i]) * dp[i] / (cp[i] * (1 - cp[i])) for i in range(n)]
        g1 = _mv(Ai, _mv(Wm, xs))
        Mid = [[WWt[i][j] - 2 * rho * WtW[i][j] for j in range(n)] for i in range(n)]
        VM = [[ssum(V[i][m] * Mid[m][j] for m in range(n)) for j in range(n)] for i in range(n)]
        d2 = [ssum(VM[i][m] * V[m][i] for m in range(n)) for i in range(n)]
        g2 = [d2[i] * xs[i] / (2 * sv[i] * sv[i]) for i in range(n)]
        du = [u[i] * (xs[i] + u[i]) for i in range(n)]
        cols = [[du[i] * AX[i][a] for i in range(n)] for a in range(k)] + [[du[i] * (g1[i] - g2[i]) for i in range(n)]]
        G = _cols([_project(Zm, c) for c in cols])
        return _ls(G, u), G, u

    it = 0
    ch = [0.0] * (k + 1)
    while True:
        it += 1
        b = [p + q for p, q in zip(b, ch)]
        ch, G, u = step(b)
        if (max(abs(v) for v in ch) <= tol and it > 1) or it >= maxit:
            break
    gg = inverse([[ssum(r[a] * r[c] for r in G) for c in range(k + 1)] for a in range(k + 1)])
    Gu = [[abs(u[i]) * G[i][a] for a in range(k + 1)] for i in range(n)]
    M = [[ssum(r[a] * r[c] for r in Gu) for c in range(k + 1)] for a in range(k + 1)]
    se = [
        math.sqrt(ssum(gg[a][c] * ssum(M[c][d] * gg[d][a] for d in range(k + 1)) for c in range(k + 1)))
        for a in range(k + 1)
    ]
    return RichResult(payload={"coefficients": b[:k], "rho": b[k], "se": se, "iterations": it})


def discrete_moran_test(y, X, W, link="probit"):
    r"""Kelejian and Prucha (2001) Moran test for spatial dependence in binary-choice models.

    From the non-spatial logit or probit fit, the generalized residuals
    ``u_i = (y_i - F_i) f_i / (F_i (1 - F_i))`` have variances ``s_i^2 = f_i^2 /
    (F_i (1 - F_i))`` (``u_i = y_i - p_i`` and ``s_i^2 = p_i (1 - p_i)`` for the
    logit); ``I = u'Wu / sqrt(sum_{i != j} (w_ij^2 + w_ij w_ji) s_i^2 s_j^2)``
    is asymptotically N(0, 1) under no spatial dependence.

    References
    ----------
    Kelejian, H. H. and Prucha, I. R. (2001). On the asymptotic distribution of
    the Moran I test statistic with applications. *Journal of Econometrics*
    104, 219-257.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [0.5, 0, 0.5, 0, 0, 0], [0, 0.5, 0, 0.5, 0, 0], [0, 0, 0.5, 0, 0.5, 0], [0, 0, 0, 0.5, 0, 0.5], [0, 0, 0, 0, 1, 0]]
    >>> X = [[1, 2.0], [1, -1.0], [1, 0.1], [1, 1.5], [1, 0.6], [1, -0.4]]
    >>> r = discrete_moran_test([1, 0, 1, 1, 0, 0], X, W, "logit")
    >>> round(r.statistic, 10)
    0.3060020248
    """
    yv, Xm, Wm = _vec(y), _mat(X), _mat(W)
    n = len(yv)
    fit = binary_glm(yv, Xm, link)
    F, eta = fit.fitted, fit.linear_predictor
    if link == "logit":
        u = [a - b for a, b in zip(yv, F)]
        s2 = [p * (1 - p) for p in F]
    else:
        f = [dnorm(e) for e in eta]
        u = [(a - b) * c / (b * (1 - b)) for a, b, c in zip(yv, F, f)]
        s2 = [c * c / (b * (1 - b)) for b, c in zip(F, f)]
    num = ssum(u[i] * ssum(Wm[i][j] * u[j] for j in range(n)) for i in range(n))
    var = ssum((Wm[i][j] ** 2 + Wm[i][j] * Wm[j][i]) * s2[i] * s2[j] for i in range(n) for j in range(n) if i != j)
    z = num / math.sqrt(var)
    return RichResult(payload={"statistic": z, "p_value": 2 * pnorm(-abs(z)), "numerator": num, "variance": var})


def cheatsheet() -> str:
    return "binary_glm / spatial_logit_gmm / spatial_probit_gmm / discrete_moran_test -> spatial discrete choice."
