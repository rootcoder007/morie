# morie.fn -- function file (rootcoder007/morie)
"""Gaussian processes (Rasmussen and Williams 2006): kernels, exact regression and marginal likelihood, prior and
posterior sample paths, hyperparameter fitting and kernel selection, closed-form leave-one-out, Laplace binary
classification, DTC and variational (VFE) sparse approximations, input warping and deep-GP prior samples."""

from __future__ import annotations

import math

from ._mvcore import cholesky
from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_normal

__all__ = [
    "gp_covariance",
    "gp_predict",
    "gp_sample",
    "gp_fit",
    "gp_loo",
    "gp_classify",
    "gp_sparse",
    "kumaraswamy_warp",
    "deep_gp_sample",
]


def _pts(a):
    return [tuple(float(v) for v in (r if isinstance(r, (list, tuple)) else [r])) for r in a]


def _k(a, b, kind, p):
    s2 = p.get("variance", 1.0)
    if kind == "linear":
        return s2 * ssum(x * y for x, y in zip(a, b)) + p.get("bias", 0.0)
    ls = p.get("lengthscale", 1.0)
    L = list(ls) if isinstance(ls, (list, tuple)) else [ls] * len(a)
    r2 = ssum(((x - y) / ell) ** 2 for x, y, ell in zip(a, b, L))
    r = math.sqrt(r2)
    if kind in ("se", "ard"):
        return s2 * math.exp(-0.5 * r2)
    if kind == "matern32":
        return s2 * (1 + math.sqrt(3) * r) * math.exp(-math.sqrt(3) * r)
    if kind == "matern52":
        return s2 * (1 + math.sqrt(5) * r + 5 * r2 / 3) * math.exp(-math.sqrt(5) * r)
    if kind == "rq":
        al = p.get("alpha", 1.0)
        return s2 * (1 + r2 / (2 * al)) ** (-al)
    if kind == "periodic":
        per = p.get("period", 1.0)
        d = math.sqrt(ssum((x - y) ** 2 for x, y in zip(a, b)))
        return s2 * math.exp(-2 * math.sin(math.pi * d / per) ** 2 / L[0] ** 2)
    raise ValueError("kernel must be se, ard, linear, periodic, matern32, matern52 or rq")


def gp_covariance(X1, X2, kernel: str = "se", **params):
    r"""Covariance matrix ``K[i][j] = k(x_i, x'_j)`` (Rasmussen and Williams 2006, chapter 4).

    ``se``/``ard`` ``s2 exp(-r^2/2)`` with ``r^2 = sum ((x - x')/l_d)^2``
    (one lengthscale, or one per dimension: automatic relevance
    determination); ``matern32`` ``s2 (1 + sqrt3 r) e^{-sqrt3 r}``;
    ``matern52`` ``s2 (1 + sqrt5 r + 5 r^2/3) e^{-sqrt5 r}``; ``rq`` ``s2 (1 +
    r^2/(2 alpha))^{-alpha}``; ``periodic`` ``s2 exp(-2 sin^2(pi |x - x'| /
    p) / l^2)``; ``linear`` ``s2 x.x' + bias``.

    References
    ----------
    Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for
    Machine Learning*. MIT Press.

    Examples
    --------
    >>> [round(v, 6) for v in gp_covariance([(0.0,)], [(0.0,), (1.0,)], "matern32")[0]]
    [1.0, 0.483358]
    """
    A, B = _pts(X1), _pts(X2)
    return [[_k(a, b, kernel, params) for b in B] for a in A]


def _chol_solve(L, b):
    n = len(L)
    z = [0.0] * n
    for i in range(n):
        z[i] = (b[i] - ssum(L[i][k] * z[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - ssum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def _fwd(L, b):
    n = len(L)
    z = [0.0] * n
    for i in range(n):
        z[i] = (b[i] - ssum(L[i][k] * z[k] for k in range(i))) / L[i][i]
    return z


def _chol(K):
    return [[float(v) for v in r] for r in cholesky(K)]


def gp_predict(X, y, Xnew, *, kernel: str = "se", noise: float = 1e-6, mean: float = 0.0, **params) -> RichResult:
    r"""Exact GP regression (Algorithm 2.1): predictive mean and variance and the log marginal likelihood.

    ``L = chol(K + s_n^2 I)``, ``alpha = L' \ (L \ (y - m))``, mean ``m + k*' alpha``,
    variance ``k(x*, x*) - |L \ k*|^2`` (latent ``f``; add ``s_n^2`` for a new
    observation), ``log p(y) = -(y - m)' alpha / 2 - sum log L_ii - n log(2 pi)/2``.

    Examples
    --------
    >>> r = gp_predict([(0.0,), (1.0,)], [0.0, 1.0], [(0.5,)], noise=0.01)
    >>> round(r.mean[0], 6), round(r.variance[0], 6)
    (0.54592, 0.036454)
    """
    P, Q = _pts(X), _pts(Xnew)
    yv = [float(v) - mean for v in y]
    n = len(P)
    K = [[_k(P[i], P[j], kernel, params) + (noise if i == j else 0.0) for j in range(n)] for i in range(n)]
    L = _chol(K)
    alpha = _chol_solve(L, yv)
    mu, var = [], []
    for q in Q:
        ks = [_k(p, q, kernel, params) for p in P]
        v = _fwd(L, ks)
        mu.append(mean + ssum(a * b for a, b in zip(ks, alpha)))
        var.append(_k(q, q, kernel, params) - ssum(x * x for x in v))
    lml = (
        -0.5 * ssum(a * b for a, b in zip(yv, alpha))
        - ssum(math.log(L[i][i]) for i in range(n))
        - n / 2 * math.log(2 * math.pi)
    )
    return RichResult(payload={"mean": mu, "variance": var, "log_marginal_likelihood": lml, "alpha": alpha})


def gp_sample(
    X,
    *,
    kernel: str = "se",
    nsim: int = 1,
    seed: int = 1,
    data=None,
    noise: float = 1e-6,
    jitter: float = 1e-10,
    **params,
) -> RichResult:
    r"""Sample paths of a GP prior (``data=None``) or of the posterior given ``data = (X_train, y_train)``.

    ``f = m + L e`` with ``L`` the Cholesky factor of the (posterior)
    covariance plus ``jitter`` on the diagonal and ``e`` standard normal
    (Philox stream ``s`` of ``seed`` for path ``s``). The posterior covariance
    is ``K** - K*' (K + s_n^2 I)^{-1} K*``.

    Examples
    --------
    >>> r = gp_sample([(0.0,), (1.0,), (2.0,)], nsim=2, seed=3)
    >>> len(r.samples), len(r.samples[0])
    (2, 3)
    """
    Q = _pts(X)
    m = len(Q)
    Kss = [[_k(a, b, kernel, params) for b in Q] for a in Q]
    mu = [0.0] * m
    if data is not None:
        P = _pts(data[0])
        yv = [float(v) for v in data[1]]
        n = len(P)
        K = [[_k(P[i], P[j], kernel, params) + (noise if i == j else 0.0) for j in range(n)] for i in range(n)]
        L = _chol(K)
        alpha = _chol_solve(L, yv)
        Ks = [[_k(p, q, kernel, params) for p in P] for q in Q]
        mu = [ssum(a * b for a, b in zip(ks, alpha)) for ks in Ks]
        V = [_fwd(L, ks) for ks in Ks]
        Kss = [[Kss[i][j] - ssum(a * b for a, b in zip(V[i], V[j])) for j in range(m)] for i in range(m)]
    C = [[Kss[i][j] + (jitter if i == j else 0.0) for j in range(m)] for i in range(m)]
    L2 = _chol(C)
    out = []
    for s in range(int(nsim)):
        e = [float(v) for v in random_normal(m, seed=seed, stream=s)]
        out.append([mu[i] + ssum(L2[i][k] * e[k] for k in range(i + 1)) for i in range(m)])
    return RichResult(payload={"samples": out, "mean": mu, "cov": Kss})


def _nelder_mead(f, x0, step=0.5, tol=1e-10, maxit=2000):
    n = len(x0)
    pts = [list(x0)] + [[x0[j] + (step if j == i else 0.0) for j in range(n)] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(maxit):
        order = sorted(range(n + 1), key=lambda i: (vals[i], i))
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + tol):
            break
        c = [ssum(p[j] for p in pts[:-1]) / n for j in range(n)]
        xr = [c[j] + (c[j] - pts[-1][j]) for j in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [c[j] + 2 * (c[j] - pts[-1][j]) for j in range(n)]
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            if fr < vals[-1]:
                xc = [c[j] + 0.5 * (xr[j] - c[j]) for j in range(n)]
            else:
                xc = [c[j] + 0.5 * (pts[-1][j] - c[j]) for j in range(n)]
            fc = f(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = [pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(n)]
                    vals[i] = f(pts[i])
    k = min(range(n + 1), key=lambda i: (vals[i], i))
    return pts[k], vals[k]


def gp_fit(X, y, *, kernels=("se",), mean: float = 0.0, init=None) -> RichResult:
    r"""Type-II maximum likelihood of GP hyperparameters (log variance, log lengthscale(s), log noise) and kernel choice.

    For each kernel in ``kernels`` the negative log marginal likelihood is
    minimised over the log hyperparameters by Nelder-Mead (deterministic
    simplex, step 0.5 from ``init`` or ``log`` of (1, 1, 0.1); periodic kernels
    also fit the log period, ``rq`` the log alpha); the kernel with the
    largest evidence is selected (Rasmussen and Williams 2006, section 5.4).
    The fitted noise variance estimates the nugget.

    Examples
    --------
    >>> r = gp_fit([(0.0,), (0.5,), (1.0,), (1.5,), (2.0,)], [0.1, 0.5, 0.8, 1.0, 0.9])
    >>> r.best
    'se'
    """
    P = _pts(X)
    yv = [float(v) for v in y]
    res = {}
    for kern in kernels:
        extra = 1 if kern in ("periodic", "rq") else 0
        x0 = list(init) if init is not None else [0.0, 0.0, math.log(0.1)] + [0.0] * extra

        def nll(th, kern=kern):
            p = {"variance": math.exp(th[0]), "lengthscale": math.exp(th[1])}
            if kern == "periodic":
                p["period"] = math.exp(th[3])
            if kern == "rq":
                p["alpha"] = math.exp(th[3])
            try:
                return -gp_predict(P, yv, [], kernel=kern, noise=math.exp(th[2]), mean=mean, **p)[
                    "log_marginal_likelihood"
                ]
            except (ValueError, ZeroDivisionError, ArithmeticError):
                return math.inf

        th, v = _nelder_mead(nll, x0)
        par = {"variance": math.exp(th[0]), "lengthscale": math.exp(th[1]), "noise": math.exp(th[2])}
        if kern == "periodic":
            par["period"] = math.exp(th[3])
        if kern == "rq":
            par["alpha"] = math.exp(th[3])
        res[kern] = {"params": par, "log_marginal_likelihood": -v}
    best = max(res, key=lambda k: res[k]["log_marginal_likelihood"])
    return RichResult(payload={"fits": res, "best": best, "params": res[best]["params"]})


def gp_loo(X, y, *, kernel: str = "se", noise: float = 1e-6, **params) -> RichResult:
    r"""Closed-form leave-one-out predictive means and variances (Rasmussen and Williams 2006, eq. 5.12).

    ``mu_i = y_i - [K^{-1} y]_i / [K^{-1}]_ii``, ``s_i^2 = 1 / [K^{-1}]_ii`` with
    ``K`` including the noise; also the LOO log predictive probability.

    Examples
    --------
    >>> r = gp_loo([(0.0,), (1.0,), (2.0,)], [0.0, 1.0, 0.0], noise=0.1)
    >>> round(r.mean[1], 6)
    0.0
    """
    P = _pts(X)
    yv = [float(v) for v in y]
    n = len(P)
    K = [[_k(P[i], P[j], kernel, params) + (noise if i == j else 0.0) for j in range(n)] for i in range(n)]
    L = _chol(K)
    alpha = _chol_solve(L, yv)
    Kinv = [_chol_solve(L, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    mu = [yv[i] - alpha[i] / Kinv[i][i] for i in range(n)]
    var = [1 / Kinv[i][i] for i in range(n)]
    lp = ssum(-0.5 * math.log(2 * math.pi * s) - (yi - m) ** 2 / (2 * s) for yi, m, s in zip(yv, mu, var))
    return RichResult(payload={"mean": mu, "variance": var, "log_predictive": lp})


def _sig(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def gp_classify(X, y, Xnew, *, kernel: str = "se", tol: float = 1e-10, maxit: int = 100, **params) -> RichResult:
    r"""Binary GP classification with the logistic likelihood by the Laplace approximation (Algorithms 3.1-3.2).

    Labels ``y`` in ``{-1, +1}``. Newton iterations ``f <- K (W f + grad) - K
    W^{1/2} B^{-1} W^{1/2} K (W f + grad)`` with ``B = I + W^{1/2} K W^{1/2}``
    until the objective changes by less than ``tol``; predictive latent mean
    ``k*' grad log p(y|f^)`` and variance ``k** - |L \ (W^{1/2} k*)|^2``, and the
    averaged class probability by MacKay's probit approximation ``sigma(mu /
    sqrt(1 + pi s^2 / 8))``. Also returns the approximate log marginal
    likelihood.

    References
    ----------
    MacKay, D. J. C. (1992). The evidence framework applied to
    classification networks. *Neural Computation*, 4(5), 720-736.

    Examples
    --------
    >>> r = gp_classify([(-1.0,), (-0.5,), (0.5,), (1.0,)], [-1, -1, 1, 1], [(0.8,)], variance=4.0)
    >>> r.probability[0] > 0.5
    True
    """
    P, Q = _pts(X), _pts(Xnew)
    yv = [float(v) for v in y]
    n = len(P)
    K = [[_k(P[i], P[j], kernel, params) for j in range(n)] for i in range(n)]
    f = [0.0] * n
    obj_old = -math.inf
    for _ in range(maxit):
        pi = [_sig(v) for v in f]
        t = [(yy + 1) / 2 for yy in yv]
        grad = [ti - p for ti, p in zip(t, pi)]
        W = [p * (1 - p) for p in pi]
        sw = [math.sqrt(w) for w in W]
        B = [[(1.0 if i == j else 0.0) + sw[i] * K[i][j] * sw[j] for j in range(n)] for i in range(n)]
        L = _chol(B)
        b = [W[i] * f[i] + grad[i] for i in range(n)]
        Kb = [ssum(K[i][j] * b[j] for j in range(n)) for i in range(n)]
        a2 = _chol_solve(L, [sw[i] * Kb[i] for i in range(n)])
        a = [b[i] - sw[i] * a2[i] for i in range(n)]
        f = [ssum(K[i][j] * a[j] for j in range(n)) for i in range(n)]
        obj = -0.5 * ssum(x * y for x, y in zip(a, f)) + ssum(-math.log1p(math.exp(-yy * v)) for yy, v in zip(yv, f))
        if abs(obj - obj_old) < tol:
            break
        obj_old = obj
    pi = [_sig(v) for v in f]
    grad = [(yy + 1) / 2 - p for yy, p in zip(yv, pi)]
    W = [p * (1 - p) for p in pi]
    sw = [math.sqrt(w) for w in W]
    B = [[(1.0 if i == j else 0.0) + sw[i] * K[i][j] * sw[j] for j in range(n)] for i in range(n)]
    L = _chol(B)
    lml = obj - ssum(math.log(L[i][i]) for i in range(n))
    mu, var, prob = [], [], []
    for q in Q:
        ks = [_k(p, q, kernel, params) for p in P]
        m = ssum(a * b for a, b in zip(ks, grad))
        v = _fwd(L, [sw[i] * ks[i] for i in range(n)])
        s2 = _k(q, q, kernel, params) - ssum(x * x for x in v)
        mu.append(m)
        var.append(s2)
        prob.append(_sig(m / math.sqrt(1 + math.pi * s2 / 8)))
    return RichResult(
        payload={
            "latent_mean": mu,
            "latent_variance": var,
            "probability": prob,
            "f_hat": f,
            "log_marginal_likelihood": lml,
        }
    )


def gp_sparse(X, y, Z, Xnew, *, method: str = "vfe", kernel: str = "se", noise: float = 0.1, **params) -> RichResult:
    r"""Sparse GP regression with inducing inputs ``Z``: DTC predictions and the Titsias (2009) VFE bound.

    With ``Q_nn = K_nm K_mm^{-1} K_mn``: the DTC / VFE predictive mean is
    ``K*m S K_mn y / s^2`` with ``S = (K_mm + K_mn K_nm / s^2)^{-1}``; the
    variance is ``k** - Q** + K*m S Km*`` (Quinonero-Candela and Rasmussen
    2005). ``log N(y | 0, Q_nn + s^2 I)`` is the DTC evidence and ``vfe``
    subtracts ``tr(K_nn - Q_nn)/(2 s^2)`` to give the variational lower bound.
    A jitter of ``1e-10`` stabilises ``K_mm``.

    References
    ----------
    Titsias, M. K. (2009). Variational learning of inducing variables in
    sparse Gaussian processes. *Proceedings of AISTATS*, 567-574.
    Quinonero-Candela, J. and Rasmussen, C. E. (2005). A unifying view of
    sparse approximate Gaussian process regression. *Journal of Machine
    Learning Research*, 6, 1939-1959.

    Examples
    --------
    >>> X = [(0.0,), (0.5,), (1.0,)]
    >>> r = gp_sparse(X, [0.0, 0.5, 1.0], X, [(0.25,)], noise=0.01)
    >>> round(r.mean[0], 3) == round(gp_predict(X, [0.0, 0.5, 1.0], [(0.25,)], noise=0.01).mean[0], 3)
    True
    """
    P, Zp, Q = _pts(X), _pts(Z), _pts(Xnew)
    yv = [float(v) for v in y]
    n, m = len(P), len(Zp)
    Kmm = [[_k(Zp[i], Zp[j], kernel, params) + (1e-10 if i == j else 0.0) for j in range(m)] for i in range(m)]
    Kmn = [[_k(Zp[i], P[j], kernel, params) for j in range(n)] for i in range(m)]
    Lm = _chol(Kmm)
    V = [_fwd(Lm, [Kmn[i][j] for i in range(m)]) for j in range(n)]  # rows: L^{-1} K_mn columns
    # Qnn = V V'
    Ssys = [[Kmm[i][j] + ssum(Kmn[i][t] * Kmn[j][t] for t in range(n)) / noise for j in range(m)] for i in range(m)]
    Ls = _chol(Ssys)
    Kmy = [ssum(Kmn[i][t] * yv[t] for t in range(n)) / noise for i in range(m)]
    w = _chol_solve(Ls, Kmy)
    mu, var = [], []
    for q in Q:
        ksm = [_k(q, z, kernel, params) for z in Zp]
        mu.append(ssum(a * b for a, b in zip(ksm, w)))
        vq = _fwd(Lm, ksm)
        vs = _fwd(Ls, ksm)
        var.append(_k(q, q, kernel, params) - ssum(x * x for x in vq) + ssum(x * x for x in vs))
    C = [[ssum(V[i][k] * V[j][k] for k in range(m)) + (noise if i == j else 0.0) for j in range(n)] for i in range(n)]
    Lc = _chol(C)
    al = _chol_solve(Lc, yv)
    dtc = (
        -0.5 * ssum(a * b for a, b in zip(yv, al))
        - ssum(math.log(Lc[i][i]) for i in range(n))
        - n / 2 * math.log(2 * math.pi)
    )
    trace = ssum(_k(P[i], P[i], kernel, params) - ssum(v * v for v in V[i]) for i in range(n))
    bound = dtc - trace / (2 * noise)
    if method not in ("vfe", "dtc"):
        raise ValueError("method must be vfe or dtc")
    return RichResult(
        payload={
            "mean": mu,
            "variance": var,
            "log_evidence": bound if method == "vfe" else dtc,
            "dtc_log_evidence": dtc,
            "vfe_bound": bound,
        }
    )


def kumaraswamy_warp(x, a: float, b: float):
    r"""Kumaraswamy CDF input warping ``w(x) = 1 - (1 - x^a)^b`` on ``[0, 1]`` (Snoek et al. 2014) for nonstationary GPs.

    References
    ----------
    Snoek, J., Swersky, K., Zemel, R. and Adams, R. P. (2014). Input warping
    for Bayesian optimization of non-stationary functions. *Proceedings of
    ICML*, 1674-1682.

    Examples
    --------
    >>> kumaraswamy_warp([0.0, 0.5, 1.0], 2.0, 1.0)
    [0.0, 0.25, 1.0]
    """
    return [1 - (1 - float(v) ** a) ** b for v in x]


def deep_gp_sample(X, layers, *, seed: int = 1, jitter: float = 1e-9) -> RichResult:
    r"""Sample from a deep-GP prior: each layer is a GP evaluated at the previous layer's output (Damianou and Lawrence 2013).

    ``layers`` is a list of ``(kernel, params)``; layer ``l`` uses Philox
    stream ``l`` of ``seed``. Returns the output of every layer.

    References
    ----------
    Damianou, A. and Lawrence, N. D. (2013). Deep Gaussian processes.
    *Proceedings of AISTATS*, 207-215.

    Examples
    --------
    >>> r = deep_gp_sample([(0.0,), (1.0,)], [("se", {}), ("se", {"lengthscale": 0.5})])
    >>> len(r.layers)
    2
    """
    cur = _pts(X)
    outs = []
    for lvl, (kern, par) in enumerate(layers):
        m = len(cur)
        K = [[_k(cur[i], cur[j], kern, par) + (jitter if i == j else 0.0) for j in range(m)] for i in range(m)]
        L = _chol(K)
        e = [float(v) for v in random_normal(m, seed=seed, stream=lvl)]
        f = [ssum(L[i][k] * e[k] for k in range(i + 1)) for i in range(m)]
        outs.append(f)
        cur = [(v,) for v in f]
    return RichResult(payload={"layers": outs, "output": outs[-1]})


def cheatsheet() -> str:
    return "gp_predict / gp_sample / gp_fit / gp_loo / gp_classify / gp_sparse -> Gaussian processes."
