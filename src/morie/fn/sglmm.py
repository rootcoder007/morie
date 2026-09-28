# morie.fn -- function file (rootcoder007/morie)
"""Spatial generalised linear mixed models: latent Gaussian fields (covariance, proper CAR, SAR and
general GMRF), response simulation (Gaussian, Poisson, binomial, negative binomial, gamma), Laplace
approximation fitting with a Gaussian posterior approximation of the latent field, predictive maps,
randomised quantile residuals and CRPS scoring."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import random_normal, random_uniform
from ._rrng_core import pnorm, qgamma, qnorm
from ._sci_core import gammaln
from .krgsys import kriging_covariance
from .lbfgsb import lbfgsb_minimize

__all__ = [
    "car_precision",
    "sar_covariance",
    "gmrf_simulate",
    "spatial_glmm_simulate",
    "glmm_laplace_fit",
    "spatial_glmm_predict",
    "glmm_residuals",
    "crps_gaussian",
    "crps_poisson",
    "crps_sample",
]


def _mat(A):
    return [[float(v) for v in r] for r in (A.tolist() if hasattr(A, "tolist") else A)]


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError("matrix is not positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _inv(A):
    return [[float(v) for v in r] for r in inverse([list(r) for r in A])]


def _logdet(A):
    return 2.0 * ssum(math.log(L) for L in (r[i] for i, r in enumerate(_chol(A))))


def car_precision(W, rho: float, tau: float = 1.0) -> list:
    r"""Precision matrix ``Q = tau (D - rho W)`` of the proper conditional autoregression (Besag 1974; Cressie 1993).

    ``W`` is a symmetric binary (or weighted) adjacency matrix and ``D`` the
    diagonal of its row sums; ``Q`` is positive definite for ``|rho| < 1``.
    Conditionally ``x_i | x_-i ~ N(rho sum_j w_ij x_j / d_i, 1 / (tau d_i))``.

    References
    ----------
    Besag, J. (1974). Spatial interaction and the statistical analysis of
    lattice systems. *JRSS B*, 36(2), 192-236.

    Examples
    --------
    >>> car_precision([[0, 1], [1, 0]], 0.5, 2.0)
    [[2.0, -1.0], [-1.0, 2.0]]
    """
    Wm = _mat(W)
    n = len(Wm)
    d = [ssum(r) for r in Wm]
    return [[tau * ((d[i] if i == j else 0.0) - rho * Wm[i][j]) for j in range(n)] for i in range(n)]


def sar_covariance(W, rho: float, sigma2: float = 1.0) -> list:
    r"""Covariance ``sigma2 [(I - rho W)'(I - rho W)]^-1`` of the simultaneous autoregression ``x = rho W x + e``.

    References
    ----------
    Whittle, P. (1954). On stationary processes in the plane. *Biometrika*,
    41(3-4), 434-449.

    Examples
    --------
    >>> [round(v, 12) for v in sar_covariance([[0, 1], [1, 0]], 0.5)[0]]
    [2.222222222222, 1.777777777778]
    """
    Wm = _mat(W)
    n = len(Wm)
    B = [[(1.0 if i == j else 0.0) - rho * Wm[i][j] for j in range(n)] for i in range(n)]
    BtB = [[ssum(B[k][i] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    return [[sigma2 * v for v in r] for r in _inv(BtB)]


def gmrf_simulate(Q, *, nsim: int = 1, seed: int = 1) -> RichResult:
    r"""Sample a zero-mean Gaussian Markov random field with precision ``Q`` (Rue and Held 2005, algorithm 2.4).

    With ``Q = L L'`` solve ``L' x = e`` for standard normal ``e`` (Philox
    stream ``s`` for sample ``s``); ``Cov(x) = Q^-1``.

    References
    ----------
    Rue, H. and Held, L. (2005). *Gaussian Markov Random Fields: Theory and
    Applications*. Chapman and Hall/CRC.

    Examples
    --------
    >>> r = gmrf_simulate([[4.0, 0.0], [0.0, 1.0]], seed=3)
    >>> from morie.fn._rng import random_normal
    >>> e = [float(v) for v in random_normal(2, seed=3, stream=0)]
    >>> [round(a - b, 12) for a, b in zip(r.samples[0], [e[0] / 2, e[1]])]
    [0.0, 0.0]
    """
    Qm = _mat(Q)
    n = len(Qm)
    L = _chol(Qm)
    out = []
    for s in range(nsim):
        e = [float(v) for v in random_normal(n, seed=seed, stream=s)]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (e[i] - ssum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
        out.append(x)
    return RichResult(payload={"samples": out})


def _draw(family, mu, u, *, trials=1, size=None, shape=None, sigma=None, z=None):
    if family == "gaussian":
        return mu + sigma * z
    if family == "poisson":
        k, p = 0, math.exp(-mu)
        F = p
        while u > F and p > 0:
            k += 1
            p *= mu / k
            F += p
        return float(k)
    if family == "binomial":
        n = int(trials)
        if mu <= 0:
            return 0.0
        if mu >= 1:
            return float(n)
        k, p = 0, (1 - mu) ** n
        F = p
        while u > F and k < n:
            p *= (n - k) / (k + 1) * mu / (1 - mu)
            k += 1
            F += p
        return float(k)
    if family == "negbin":
        r = float(size)
        k, p = 0, math.exp(r * math.log(r / (r + mu)))
        F = p
        while u > F and p > 0:
            p *= (k + r) / (k + 1) * mu / (r + mu)
            k += 1
            F += p
        return float(k)
    if family == "gamma":
        return float(qgamma(u, shape, shape / mu))
    raise ValueError("family must be gaussian, poisson, binomial, negbin or gamma")


def spatial_glmm_simulate(
    X,
    beta,
    latent,
    *,
    family: str = "poisson",
    trials=1,
    size: float | None = None,
    shape: float | None = None,
    sigma: float = 1.0,
    seed: int = 1,
) -> RichResult:
    r"""Simulate responses of a spatial GLMM ``g(E y_i) = x_i'beta + S_i`` given a latent field realisation ``S``.

    Families and links: ``gaussian`` (identity, noise sd ``sigma``),
    ``poisson`` (log), ``binomial`` (logit, ``trials``), ``negbin`` (log,
    NB2 with ``size``, variance ``mu + mu^2/size``), ``gamma`` (log,
    ``shape``, mean ``mu``). Draws by inversion of Philox uniforms (stream
    0; Gaussian noise stream 1) with exact pmf recurrences, so the draws
    are reproducible across arms. The latent field can come from
    :func:`~morie.fn.zschl.chol_sim` (geostatistical), :func:`gmrf_simulate`
    with :func:`car_precision` (CAR) or a Cholesky factor of
    :func:`sar_covariance` (SAR) (Diggle, Tawn and Moyeed 1998).

    References
    ----------
    Diggle, P. J., Tawn, J. A. and Moyeed, R. A. (1998). Model-based
    geostatistics. *Applied Statistics*, 47(3), 299-350.

    Examples
    --------
    >>> r = spatial_glmm_simulate([[1.0], [1.0]], [0.0], [0.0, 0.0], family="binomial", trials=1)
    >>> r.mu
    [0.5, 0.5]
    """
    Xm = _mat(X)
    b = [float(v) for v in beta]
    S = [float(v) for v in latent]
    n = len(Xm)
    eta = [ssum(Xm[i][a] * b[a] for a in range(len(b))) + S[i] for i in range(n)]
    if family == "binomial":
        mu = [1 / (1 + math.exp(-e)) for e in eta]
    elif family == "gaussian":
        mu = eta
    else:
        mu = [math.exp(e) for e in eta]
    u = [float(v) for v in random_uniform(n, seed=seed, stream=0)]
    z = [float(v) for v in random_normal(n, seed=seed, stream=1)] if family == "gaussian" else [None] * n
    tr = [trials] * n if not hasattr(trials, "__len__") else list(trials)
    y = [_draw(family, mu[i], u[i], trials=tr[i], size=size, shape=shape, sigma=sigma, z=z[i]) for i in range(n)]
    return RichResult(payload={"y": y, "eta": eta, "mu": mu, "family": family})


def _ll_terms(family, y, eta, m):
    if family == "poisson":
        mu = math.exp(eta)
        return y * eta - mu - gammaln(y + 1), y - mu, mu
    p = 1 / (1 + math.exp(-eta))
    ll = y * eta - m * math.log1p(math.exp(eta)) if eta < 30 else y * eta - m * (eta + math.log1p(math.exp(-eta)))
    ll += gammaln(m + 1) - gammaln(y + 1) - gammaln(m - y + 1)
    return ll, y - m * p, m * p * (1 - p)


def _laplace(y, X, Z, Sigma, beta, family, m, u0):
    """Inner Newton for the posterior mode of u and the Laplace log-likelihood."""
    n, q = len(y), len(Sigma)
    Si = _inv(Sigma)
    off = [ssum(X[i][a] * beta[a] for a in range(len(beta))) for i in range(n)]
    u = list(u0)
    for _ in range(100):
        eta = [off[i] + ssum(Z[i][k] * u[k] for k in range(q)) for i in range(n)]
        terms = [_ll_terms(family, y[i], eta[i], m[i]) for i in range(n)]
        g = [ssum(Z[i][k] * terms[i][1] for i in range(n)) - ssum(Si[k][j] * u[j] for j in range(q)) for k in range(q)]
        H = [[Si[a][b] + ssum(Z[i][a] * terms[i][2] * Z[i][b] for i in range(n)) for b in range(q)] for a in range(q)]
        Hi = _inv(H)
        step = [ssum(Hi[a][b] * g[b] for b in range(q)) for a in range(q)]
        u = [a + s for a, s in zip(u, step)]
        if max(abs(s) for s in step) < 1e-12 * (1 + max(abs(v) for v in u)):
            break
    eta = [off[i] + ssum(Z[i][k] * u[k] for k in range(q)) for i in range(n)]
    terms = [_ll_terms(family, y[i], eta[i], m[i]) for i in range(n)]
    H = [[Si[a][b] + ssum(Z[i][a] * terms[i][2] * Z[i][b] for i in range(n)) for b in range(q)] for a in range(q)]
    quad = ssum(u[a] * Si[a][b] * u[b] for a in range(q) for b in range(q))
    ll = ssum(t[0] for t in terms) - 0.5 * quad - 0.5 * (_logdet(Sigma) + _logdet(H))
    return ll, u, H, eta


def glmm_laplace_fit(
    y, X, *, family: str = "poisson", coords=None, groups=None, model: str = "Exp", trials=None, start=None
) -> RichResult:
    r"""Fit a GLMM with Gaussian random effects by maximising the Laplace approximation of the marginal likelihood.

    Random effects: iid intercepts per level of ``groups`` (``Sigma =
    sigma2 I``; the model of ``lme4::glmer(y ~ X + (1 | g))`` with
    ``nAGQ = 1``) or a spatial field at ``coords`` with ``Sigma = sigma2
    rho(d / range)`` (``model`` ``Exp``, ``Gau`` or ``Sph``; Diggle et al.
    1998). For fixed parameters the latent mode ``u`` is found by Newton's
    method, and ``log L = log p(y | u) - u'Sigma^-1 u / 2 - (log|Sigma| +
    log|Sigma^-1 + Z'WZ|) / 2``; ``beta``, ``log sigma`` (and ``log range``)
    are maximised by L-BFGS-B. Families ``poisson`` (log) and ``binomial``
    (logit, ``trials``). The Gaussian approximation of the latent posterior
    (the Laplace/INLA building block; Rue, Martino and Chopin 2009) is
    returned as ``u`` (mode) and ``posterior_sd`` (from ``(Sigma^-1 +
    Z'WZ)^-1``).

    References
    ----------
    Breslow, N. E. and Clayton, D. G. (1993). Approximate inference in
    generalized linear mixed models. *JASA*, 88(421), 9-25.
    Rue, H., Martino, S. and Chopin, N. (2009). Approximate Bayesian
    inference for latent Gaussian models by using integrated nested Laplace
    approximations. *JRSS B*, 71(2), 319-392.

    Examples
    --------
    >>> r = glmm_laplace_fit([1, 3, 2, 6, 4, 9], [[1]] * 6, groups=[0, 0, 1, 1, 2, 2])
    >>> r.sigma2 > 0 and len(r.u) == 3
    True
    """
    yv = [float(v) for v in y]
    Xm = _mat(X)
    n, p = len(yv), len(Xm[0])
    m = (
        [1.0] * n
        if trials is None
        else ([float(trials)] * n if not hasattr(trials, "__len__") else [float(v) for v in trials])
    )
    if groups is not None:
        levels = sorted(set(groups), key=list(groups).index)
        Z = [[1.0 if g == lv else 0.0 for lv in levels] for g in groups]
        q = len(levels)
        D = None
    elif coords is not None:
        P = [tuple(float(v) for v in r) for r in coords]
        Z = [[1.0 if i == k else 0.0 for k in range(n)] for i in range(n)]
        q = n
        D = [[math.dist(P[i], P[j]) for j in range(n)] for i in range(n)]
    else:
        raise ValueError("give groups or coords")

    def sigma_of(theta):
        s2 = math.exp(2 * theta[0])
        if D is None:
            return [[s2 if a == b else 0.0 for b in range(q)] for a in range(q)]
        comp = {"model": model, "psill": s2, "range": math.exp(theta[1])}
        return [
            [kriging_covariance(D[a][b], comp) + (1e-10 * s2 if a == b else 0.0) for b in range(q)] for a in range(q)
        ]

    ntheta = 1 if D is None else 2
    if start is None:
        mbar = ssum(yv) / ssum(m)
        link = math.log(max(mbar, 1e-8)) if family == "poisson" else math.log(max(mbar, 1e-8) / max(1 - mbar, 1e-8))
        start = [link] + [0.0] * (p - 1) + [math.log(0.5)] + ([math.log(ssum(r[-1] for r in D) / n)] if D else [])
    state = {"u": [0.0] * q}

    def obj(par):
        beta, theta = par[:p], par[p:]
        ll, u, _, _ = _laplace(yv, Xm, Z, sigma_of(theta), beta, family, m, state["u"])
        state["u"] = u
        return -ll

    res = lbfgsb_minimize(obj, [float(v) for v in start], pgtol=1e-7, factr=1e3, max_iter=500)
    par = [float(v) for v in res.x]
    beta, theta = par[:p], par[p:]
    Sigma = sigma_of(theta)
    ll, u, H, eta = _laplace(yv, Xm, Z, Sigma, beta, family, m, state["u"])
    V = _inv(H)
    k = p + ntheta
    return RichResult(
        payload={
            "beta": beta,
            "sigma2": math.exp(2 * theta[0]),
            "range": math.exp(theta[1]) if D is not None else None,
            "loglik": ll,
            "aic": -2 * ll + 2 * k,
            "u": u,
            "posterior_sd": [math.sqrt(V[a][a]) for a in range(q)],
            "posterior_cov": V,
            "eta": eta,
            "family": family,
            "model": model,
            "converged": res.converged,
            "coords": coords,
            "trials": m,
        }
    )


def _gh(n=20):
    """Gauss-Hermite nodes and weights for the weight exp(-x^2) (Newton on orthonormal Hermite polynomials)."""
    x, w = [0.0] * n, [0.0] * n
    z = 0.0
    for i in range((n + 1) // 2):
        if i == 0:
            z = math.sqrt(2 * n + 1) - 1.85575 * (2 * n + 1) ** (-1 / 6)
        elif i == 1:
            z -= 1.14 * n**0.426 / z
        elif i == 2:
            z = 1.86 * z - 0.86 * x[0]
        elif i == 3:
            z = 1.91 * z - 0.91 * x[1]
        else:
            z = 2 * z - x[i - 2]
        pp = 1.0
        for _ in range(100):
            p1, p2 = math.pi**-0.25, 0.0
            for j in range(1, n + 1):
                p3, p2 = p2, p1
                p1 = z * math.sqrt(2 / j) * p2 - math.sqrt((j - 1) / j) * p3
            pp = math.sqrt(2 * n) * p2
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) <= 1e-15:
                break
        x[i], x[n - 1 - i] = z, -z
        w[i] = w[n - 1 - i] = 2 / (pp * pp)
    return x, w


def spatial_glmm_predict(fit, X0, coords0) -> RichResult:
    r"""Predictive map of a fitted spatial GLMM (:func:`glmm_laplace_fit` with ``coords``).

    The latent field at new sites is kriged from the posterior mode:
    ``m* = c'Sigma^-1 u`` and ``v* = sigma2 - c'Sigma^-1 c + c'Sigma^-1 V
    Sigma^-1 c`` with ``V`` the posterior covariance of ``u``; the linear
    predictor ``x0'beta + S*`` is Gaussian and the response mean is
    ``exp(eta + v/2)`` (Poisson) or the logit-normal mean by 20-point
    Gauss-Hermite quadrature (binomial, per trial).

    References
    ----------
    Diggle, P. J. and Ribeiro, P. J. (2007). *Model-based Geostatistics*.
    Springer.

    Examples
    --------
    >>> f = glmm_laplace_fit([2, 4, 3, 7, 5], [[1]] * 5, coords=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    >>> p = spatial_glmm_predict(f, [[1]], [(1, 0)])
    >>> abs(p.eta[0] - f.eta[1]) < 1e-6
    True
    """
    P = [tuple(float(v) for v in r) for r in fit.coords]
    Q = [tuple(float(v) for v in r) for r in coords0]
    comp = {"model": fit.model, "psill": fit.sigma2, "range": fit.range}
    n = len(P)
    Sigma = [
        [kriging_covariance(math.dist(P[a], P[b]), comp) + (1e-10 * fit.sigma2 if a == b else 0.0) for b in range(n)]
        for a in range(n)
    ]
    Si = _inv(Sigma)
    Siu = [ssum(Si[a][b] * fit.u[b] for b in range(n)) for a in range(n)]
    V = fit.posterior_cov
    eta, var, mean = [], [], []
    xs, ws = _gh()
    for x0, q in zip(X0, Q):
        c = [kriging_covariance(math.dist(P[a], q), comp) for a in range(n)]
        w = [ssum(Si[a][b] * c[b] for b in range(n)) for a in range(n)]
        mstar = ssum(c[a] * Siu[a] for a in range(n))
        v = (
            fit.sigma2
            - ssum(c[a] * w[a] for a in range(n))
            + ssum(w[a] * V[a][b] * w[b] for a in range(n) for b in range(n))
        )
        e = ssum(float(x0[a]) * fit.beta[a] for a in range(len(fit.beta))) + mstar
        eta.append(e)
        var.append(v)
        if fit.family == "poisson":
            mean.append(math.exp(e + v / 2))
        else:
            sd = math.sqrt(max(v, 0.0))
            mean.append(
                ssum(wk * 1 / (1 + math.exp(-(e + math.sqrt(2) * sd * xk))) for xk, wk in zip(xs, ws))
                / math.sqrt(math.pi)
            )
    return RichResult(payload={"eta": eta, "variance": var, "mean": mean})


def glmm_residuals(y, mu, *, family: str = "poisson", trials=1, size: float | None = None, seed: int = 1) -> RichResult:
    r"""Pearson, deviance and randomised quantile residuals (Dunn and Smyth 1996) of a count GLMM.

    Randomised quantile residuals ``Phi^-1(F(y - 1) + v (F(y) - F(y - 1)))``
    with Philox uniforms ``v`` are exactly standard normal under the model
    (the basis of simulation-free GLMM diagnostics, cf. DHARMa). Families
    ``poisson``, ``binomial`` (``mu`` a probability, ``trials``) and
    ``negbin`` (``size``).

    References
    ----------
    Dunn, P. K. and Smyth, G. K. (1996). Randomized quantile residuals.
    *Journal of Computational and Graphical Statistics*, 5(3), 236-244.

    Examples
    --------
    >>> [round(v, 12) for v in glmm_residuals([2, 0], [2.0, 1.0]).pearson]
    [0.0, -1.0]
    """
    yv = [float(v) for v in y]
    mv = [float(v) for v in mu]
    n = len(yv)
    tr = [float(trials)] * n if not hasattr(trials, "__len__") else [float(v) for v in trials]
    v = [float(x) for x in random_uniform(n, seed=seed, stream=0)]

    def cdf(k, i):
        if k < 0:
            return 0.0
        mu_ = mv[i]
        if family == "poisson":
            p = math.exp(-mu_)
            F = p
            for j in range(1, int(k) + 1):
                p *= mu_ / j
                F += p
            return min(F, 1.0)
        if family == "binomial":
            nn = int(tr[i])
            p = (1 - mu_) ** nn
            F = p
            for j in range(int(k)):
                p *= (nn - j) / (j + 1) * mu_ / (1 - mu_)
                F += p
            return min(F, 1.0)
        r = float(size)
        p = math.exp(r * math.log(r / (r + mu_)))
        F = p
        for j in range(int(k)):
            p *= (j + r) / (j + 1) * mu_ / (r + mu_)
            F += p
        return min(F, 1.0)

    pear, dev, rq = [], [], []
    for i in range(n):
        yi, mi = yv[i], mv[i]
        if family == "poisson":
            var = mi
            d = 2 * ((yi * math.log(yi / mi) if yi > 0 else 0.0) - (yi - mi))
            mean = mi
        elif family == "binomial":
            nn = tr[i]
            mean, var = nn * mi, nn * mi * (1 - mi)
            d = 2 * (
                (yi * math.log(yi / mean) if yi > 0 else 0.0)
                + ((nn - yi) * math.log((nn - yi) / (nn - mean)) if nn - yi > 0 else 0.0)
            )
        else:
            r = float(size)
            mean, var = mi, mi + mi * mi / r
            d = 2 * ((yi * math.log(yi / mi) if yi > 0 else 0.0) - (yi + r) * math.log((yi + r) / (mi + r)))
        pear.append((yi - mean) / math.sqrt(var))
        dev.append(math.copysign(math.sqrt(max(d, 0.0)), yi - mean))
        a, b = cdf(yi - 1, i), cdf(yi, i)
        uu = min(max(a + v[i] * (b - a), 1e-300), 1 - 1e-16)
        rq.append(float(qnorm(uu)))
    return RichResult(payload={"pearson": pear, "deviance": dev, "quantile": rq})


def crps_gaussian(y, mu, sigma) -> list:
    r"""CRPS of Gaussian predictive distributions, ``sigma (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))``, ``z = (y - mu)/sigma`` (Gneiting et al. 2005).

    References
    ----------
    Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. *JASA*, 102(477), 359-378.

    Examples
    --------
    >>> round(crps_gaussian([0.0], [0.0], [1.0])[0], 12)
    0.233694977255
    """
    out = []
    for a, m, s in zip(y, mu, sigma):
        z = (float(a) - float(m)) / float(s)
        out.append(
            float(s)
            * (
                z * (2 * float(pnorm(z)) - 1)
                + 2 * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
                - 1 / math.sqrt(math.pi)
            )
        )
    return out


def crps_poisson(y, lam) -> list:
    r"""CRPS of Poisson predictive distributions, ``sum_k (F(k) - 1(y <= k))^2`` summed past ``max(y, lambda)`` until the pmf is below ``1e-20``.

    Equal to the closed form of Wei and Held (2014) used by
    ``scoringRules::crps_pois``.

    References
    ----------
    Wei, W. and Held, L. (2014). Calibration tests for count data. *Test*,
    23(4), 787-805.

    Examples
    --------
    >>> round(crps_poisson([0], [0.5])[0], 12) == round(sum((1 - math.exp(-0.5) * sum(0.5 ** j / math.factorial(j) for j in range(k + 1))) ** 2 for k in range(60)), 12)
    True
    """
    out = []
    for a, l_ in zip(y, lam):
        a, l_ = float(a), float(l_)
        p = math.exp(-l_)
        F = p
        k, s = 0, 0.0
        while True:
            s += (F - (1.0 if a <= k else 0.0)) ** 2
            if k >= max(a, l_) and p < 1e-20:
                break
            k += 1
            p *= l_ / k
            F += p
        out.append(s)
    return out


def crps_sample(y, samples) -> list:
    r"""Sample-based CRPS ``mean |X - y| - mean |X - X'| / 2`` over the empirical predictive distribution (as ``scoringRules::crps_sample``).

    Examples
    --------
    >>> crps_sample([1.0], [[0.0, 2.0]])
    [0.5]
    """
    out = []
    for a, s in zip(y, samples):
        s = [float(v) for v in s]
        m = len(s)
        out.append(ssum(abs(v - float(a)) for v in s) / m - ssum(abs(u - v) for u in s for v in s) / (2 * m * m))
    return out


def cheatsheet() -> str:
    return (
        "car_precision / sar_covariance / gmrf_simulate / spatial_glmm_simulate / glmm_laplace_fit / "
        "spatial_glmm_predict / glmm_residuals / crps_gaussian / crps_poisson / crps_sample -> spatial GLMMs."
    )
