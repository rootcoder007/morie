# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel data models (Elhorst 2014): fixed-effects spatial lag, error and Durbin models
by concentrated maximum likelihood, the random-effects spatial lag model, and the dynamic
spatial panel (spatial ARX) with time- and space-time-lagged responses."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult
from .lbfgsb import lbfgsb_minimize

__all__ = ["sp_panel_fe", "sp_panel_re", "sp_panel_dynamic"]


def _matvec(M, v):
    return [ssum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _eigs(W):
    return [complex(v) for v in np.linalg.eigvals(np.asarray(W, dtype=float))]


def _ld_eig(ev, p):
    # ln|I - p W| = sum ln|1 - p lambda_i| (Ord 1975)
    return ssum(math.log(abs(1.0 - p * v)) for v in ev)


def _tr_eig(ev, p):
    # tr(W (I - p W)^(-1)) = sum lambda_i / (1 - p lambda_i)
    return ssum((v / (1.0 - p * v)).real for v in ev)


def _refine(dfun, x0, lo, hi, width=1e-6):
    # bisection on the score in a small bracket around the golden-section optimum
    a, b = max(lo, x0 - width), min(hi, x0 + width)
    fa, fb = dfun(a), dfun(b)
    if fa * fb > 0:
        return x0
    for _ in range(100):
        m = (a + b) / 2.0
        if m in (a, b):
            break
        fm = dfun(m)
        if fm == 0.0:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return (a + b) / 2.0


def _ols(X, y):
    k = len(X[0])
    XtX = [[ssum(r[a] * r[b] for r in X) for b in range(k)] for a in range(k)]
    Xty = [ssum(r[a] * v for r, v in zip(X, y)) for a in range(k)]
    b = solve(XtX, Xty)
    e = [v - ssum(r[j] * b[j] for j in range(k)) for r, v in zip(X, y)]
    return b, e, XtX


def _demean(v, T, N, effects):
    # v stacked time-major: v[t * N + i]
    out = list(v)
    if effects in ("individual", "twoways"):
        um = [ssum(v[t * N + i] for t in range(T)) / T for i in range(N)]
        out = [out[t * N + i] - um[i] for t in range(T) for i in range(N)]
    if effects in ("time", "twoways"):
        tm = [ssum(v[t * N + i] for i in range(N)) / N for t in range(T)]
        out = [out[t * N + i] - tm[t] for t in range(T) for i in range(N)]
        if effects == "twoways":
            g = ssum(v) / (T * N)
            out = [a + g for a in out]
    return out


def _golden(f, lo, hi, tol=1e-12, max_iter=300):
    r = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - r * (b - a), a + r * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if b - a <= tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - r * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + r * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def _stack(y, X, W, lagx):
    T, N = len(y), len(y[0])
    K = len(X[0][0]) if X is not None and len(X[0][0]) else 0
    ys = [float(y[t][i]) for t in range(T) for i in range(N)]
    wy = [v for t in range(T) for v in _matvec(W, [float(a) for a in y[t]])]
    cols = [[float(X[t][i][k]) for t in range(T) for i in range(N)] for k in range(K)]
    if lagx:
        for k in range(K):
            cols.append([v for t in range(T) for v in _matvec(W, [float(X[t][i][k]) for i in range(N)])])
    return T, N, ys, wy, cols


def sp_panel_fe(
    y,
    X,
    W,
    *,
    model: str = "lag",
    effects: str = "individual",
    bounds=(-0.99, 0.99),
    tol: float = 1e-8,
    method: str = "elhorst",
) -> RichResult:
    r"""Fixed-effects spatial panel models by concentrated maximum likelihood (Elhorst 2014, ch. 3).

    ``y[t][i]`` (``T x N``), ``X[t][i][k]`` and weights ``W`` (``N x N``).
    Data are demeaned for ``effects`` ("individual", "time" or "twoways";
    ``Wy`` is demeaned directly). ``model="lag"``:
    ``y = rho W y + X beta + mu + e``; with ``e_0, e_1`` the residuals of
    ``y~`` and ``(Wy)~`` on ``X~``, ``rho`` maximises
    ``-NT/2 ln((e_0 - rho e_1)'(e_0 - rho e_1)) + T ln|I - rho W|`` (golden
    section on ``bounds``, then bisection on the score in a 1e-6 bracket so the
    optimum is found to rounding level; log-determinants and traces from the
    eigenvalues of ``W``, Ord 1975) and ``beta = b_0 - rho b_1``. ``(Wy)~`` is ``Wy``
    demeaned (Elhorst); ``method="splm"`` uses ``W`` times the demeaned ``y``
    instead, as ``splm::spml`` does (identical for individual effects). ``model="durbin"``
    adds ``WX`` to the regressors. ``model="error"``:
    ``u = lambda W u + e``, ``lambda`` maximising
    ``-NT/2 ln(e(lambda)'e(lambda)) + T ln|I - lambda W|`` with ``beta`` by
    least squares on ``y~ - lambda W y~`` and ``X~ - lambda W X~``. Standard
    errors come from the inverse information matrix (with
    ``W~ = W (I - rho W)^(-1)``).

    References
    ----------
    Elhorst, J. P. (2014). Spatial Econometrics: From Cross-Sectional Data to
    Spatial Panels. Springer, ch. 3. Elhorst, J. P. (2003). Specification and
    estimation of spatial panel data models. Int. Regional Sci. Rev. 26, 244-268.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> y = [[1.0, 2.0, 1.5], [2.0, 2.5, 1.0], [1.5, 3.0, 2.5], [2.5, 2.0, 3.0]]
    >>> X = [[[0.5], [1.0], [0.2]], [[1.5], [0.7], [0.1]], [[0.4], [2.0], [1.1]], [[1.2], [0.3], [1.9]]]
    >>> r = sp_panel_fe(y, X, W)
    >>> round(r.rho, 6), [round(b, 6) for b in r.beta]
    (0.113355, [0.866506])
    """
    durbin = model == "durbin"
    T, N, ys, wy, cols = _stack(y, X, W, durbin)
    NT = T * N
    yt = _demean(ys, T, N, effects)
    Xt = [_demean(c, T, N, effects) for c in cols]
    K = len(Xt)
    Xm = [[Xt[k][r] for k in range(K)] for r in range(NT)]
    Wf = [[float(v) for v in row] for row in W]

    def wstack(v):
        return [a for t in range(T) for a in _matvec(Wf, v[t * N : (t + 1) * N])]

    ev = _eigs(Wf)

    def ld(p):
        return _ld_eig(ev, p)

    if model in ("lag", "durbin"):
        wyt = wstack(yt) if method == "splm" else _demean(wy, T, N, effects)
        b0, e0, XtX = _ols(Xm, yt)
        b1, e1, _ = _ols(Xm, wyt)

        def negll(p):
            ee = ssum((a - p * b) ** 2 for a, b in zip(e0, e1))
            return 0.5 * NT * math.log(ee) - T * ld(p)

        def score(p):
            r = [a - p * b for a, b in zip(e0, e1)]
            return NT * ssum(b * a for a, b in zip(r, e1)) / ssum(a * a for a in r) - T * _tr_eig(ev, p)

        rho = _refine(score, _golden(negll, bounds[0], bounds[1], tol), bounds[0], bounds[1])
        beta = [a - rho * b for a, b in zip(b0, b1)]
        e = [a - rho * b for a, b in zip(e0, e1)]
        s2 = ssum(v * v for v in e) / NT
        A = [[(1.0 if i == j else 0.0) - rho * Wf[i][j] for j in range(N)] for i in range(N)]
        Ai = inverse(A)
        Wt = [[ssum(Wf[i][m] * Ai[m][j] for m in range(N)) for j in range(N)] for i in range(N)]
        xb = [ssum(Xm[r][k] * beta[k] for k in range(K)) for r in range(NT)]
        wxb = [a for t in range(T) for a in _matvec(Wt, xb[t * N : (t + 1) * N])]
        trW = ssum(Wt[i][i] for i in range(N))
        trWW = ssum(Wt[i][j] * Wt[j][i] + Wt[i][j] * Wt[i][j] for i in range(N) for j in range(N))
        P = K + 2
        Info = [[0.0] * P for _ in range(P)]
        for a in range(K):
            for b in range(K):
                Info[a][b] = XtX[a][b] / s2
            Info[a][K] = Info[K][a] = ssum(Xm[r][a] * wxb[r] for r in range(NT)) / s2
        Info[K][K] = T * trWW + ssum(v * v for v in wxb) / s2
        Info[K][K + 1] = Info[K + 1][K] = T * trW / s2
        Info[K + 1][K + 1] = NT / (2 * s2 * s2)
        V = inverse(Info)
        ll = -0.5 * NT * (math.log(2 * math.pi * s2) + 1) + T * ld(rho)
        return RichResult(
            payload={
                "rho": rho,
                "beta": beta,
                "sigma2": s2,
                "loglik": ll,
                "se_beta": [math.sqrt(V[k][k]) for k in range(K)],
                "se_rho": math.sqrt(V[K][K]),
                "residuals": e,
                "model": model,
                "effects": effects,
            }
        )
    wyt_all = wstack(yt)
    wX = [wstack(c) for c in Xt]

    def fit(lam):
        ys_ = [a - lam * b for a, b in zip(yt, wyt_all)]
        Xs = [[Xt[k][r] - lam * wX[k][r] for k in range(K)] for r in range(NT)]
        b, e, XtX = _ols(Xs, ys_)
        return b, e, XtX

    def negll_e(lam):
        e = fit(lam)[1]
        return 0.5 * NT * math.log(ssum(v * v for v in e)) - T * ld(lam)

    def score_e(lam):
        b, e, _ = fit(lam)
        d = [wyt_all[r] - ssum(wX[k][r] * b[k] for k in range(K)) for r in range(NT)]
        return NT * ssum(a * c for a, c in zip(e, d)) / ssum(a * a for a in e) - T * _tr_eig(ev, lam)

    lam = _refine(score_e, _golden(negll_e, bounds[0], bounds[1], tol), bounds[0], bounds[1])
    beta, e, XtX = fit(lam)
    s2 = ssum(v * v for v in e) / NT
    A = [[(1.0 if i == j else 0.0) - lam * Wf[i][j] for j in range(N)] for i in range(N)]
    Ai = inverse(A)
    Bt = [[ssum(Wf[i][m] * Ai[m][j] for m in range(N)) for j in range(N)] for i in range(N)]
    trB = ssum(Bt[i][i] for i in range(N))
    trBB = ssum(Bt[i][j] * Bt[j][i] + Bt[i][j] * Bt[i][j] for i in range(N) for j in range(N))
    Vb = inverse([[XtX[a][b] / s2 for b in range(K)] for a in range(K)])
    V2 = inverse([[T * trBB, T * trB / s2], [T * trB / s2, NT / (2 * s2 * s2)]])
    ll = -0.5 * NT * (math.log(2 * math.pi * s2) + 1) + T * ld(lam)
    return RichResult(
        payload={
            "lambda_": lam,
            "beta": beta,
            "sigma2": s2,
            "loglik": ll,
            "se_beta": [math.sqrt(Vb[k][k]) for k in range(K)],
            "se_lambda": math.sqrt(V2[0][0]),
            "residuals": e,
            "model": model,
            "effects": effects,
        }
    )


def sp_panel_re(y, X, W, *, bounds=(-0.99, 0.99), max_iter: int = 500) -> RichResult:
    r"""Random-effects spatial lag panel ``(I_T x A) y = X beta + (iota_T x I_N) mu + e`` by ML.

    ``A = I - rho W``; with ``phi = sigma_mu^2 / sigma^2`` the error
    covariance is ``sigma^2 (phi J_T x I_N + I_NT)`` and quasi-demeaning by
    ``1 - 1/sqrt(1 + T phi)`` whitens it. ``beta`` and ``sigma^2`` are
    concentrated out (``X`` gets an intercept column) and
    ``-NT/2 ln(sigma^2) - N/2 ln(1 + T phi) + T ln|A|`` is maximised over
    ``(phi, rho)`` by L-BFGS-B with the analytic (envelope) score (``phi >= 0``,
    ``rho`` in ``bounds``), polished by Newton steps on the score.

    References
    ----------
    Elhorst, J. P. (2014). Spatial Econometrics, section 3.4. Millo, G. and
    Piras, G. (2012). splm: spatial panel data models in R. J. Stat.
    Software 47(1). Baltagi, B. H., Song, S. H. and Koh, W. (2003). J.
    Econometrics 117, 123-150.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> y = [[1.0, 2.0, 1.5], [2.0, 2.5, 1.0], [1.5, 3.0, 2.5], [2.5, 2.0, 3.0]]
    >>> X = [[[0.5], [1.0], [0.2]], [[1.5], [0.7], [0.1]], [[0.4], [2.0], [1.1]], [[1.2], [0.3], [1.9]]]
    >>> r = sp_panel_re(y, X, W)
    >>> r.phi >= 0
    True
    """
    T, N, ys, wy, cols = _stack(y, X, W, False)
    NT = T * N
    K = len(cols)
    Xr = [[1.0] + [cols[k][r] for k in range(K)] for r in range(NT)]

    def qd(v, th):
        um = [ssum(v[t * N + i] for t in range(T)) / T for i in range(N)]
        return [v[t * N + i] - th * um[i] for t in range(T) for i in range(N)]

    Wf = [[float(v) for v in row] for row in W]

    ev = _eigs(Wf)

    def conc(par, grad=False):
        phi, rho = par
        th = 1.0 - 1.0 / math.sqrt(1.0 + T * phi)
        raw = [a - rho * b for a, b in zip(ys, wy)]
        ay = qd(raw, th)
        Xq = [qd([Xr[r][k] for r in range(NT)], th) for k in range(K + 1)]
        Xs = [[Xq[k][r] for k in range(K + 1)] for r in range(NT)]
        b, e, _ = _ols(Xs, ay)
        ee = ssum(v * v for v in e)
        s2 = ee / NT
        ll = -0.5 * NT * (math.log(2 * math.pi * s2) + 1) - 0.5 * N * math.log(1 + T * phi) + T * _ld_eig(ev, rho)
        if not grad:
            return ll, b, s2
        # envelope-theorem score: d e / d rho = -q(W y), d e / d theta = -(unit means of y - rho W y - X b)
        qwy = qd(wy, th)
        g_rho = NT * ssum(a * c for a, c in zip(e, qwy)) / ee - T * _tr_eig(ev, rho)
        rr = [raw[r] - ssum(Xr[r][k] * b[k] for k in range(K + 1)) for r in range(NT)]
        um = [ssum(rr[t * N + i] for t in range(T)) / T for i in range(N)]
        de_dth = [-um[i] for t in range(T) for i in range(N)]
        dth = 0.5 * T * (1.0 + T * phi) ** -1.5
        g_phi = -NT * ssum(a * c for a, c in zip(e, de_dth)) / ee * dth - 0.5 * N * T / (1 + T * phi)
        return ll, [g_phi, g_rho]

    def f(par):
        return -conc(par)[0]

    def g(par):
        return [-v for v in conc(par, True)[1]]

    res = lbfgsb_minimize(
        f,
        [0.5, 0.0],
        grad=g,
        lower=[0.0, bounds[0]],
        upper=[1e6, bounds[1]],
        pgtol=1e-12,
        factr=1.0,
        max_iter=max_iter,
    )
    x = [float(res.x[0]), float(res.x[1])]
    # Newton polish on the analytic score (Hessian by central differences of the score)
    for _ in range(20):
        gx = g(x)
        H = []
        for a in range(2):
            h = 1e-6 * max(1.0, abs(x[a]))
            xp, xm = list(x), list(x)
            xp[a] += h
            xm[a] -= h
            gp, gm = g(xp), g(xm)
            H.append([(gp[b] - gm[b]) / (2 * h) for b in range(2)])
        H = [[(H[a][b] + H[b][a]) / 2 for b in range(2)] for a in range(2)]
        det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
        if det <= 0 or H[0][0] <= 0:
            break
        step = [(H[1][1] * gx[0] - H[0][1] * gx[1]) / det, (H[0][0] * gx[1] - H[1][0] * gx[0]) / det]
        new = [min(max(x[0] - step[0], 0.0), 1e6), min(max(x[1] - step[1], bounds[0]), bounds[1])]
        done = max(abs(new[a] - x[a]) / max(1.0, abs(x[a])) for a in range(2)) <= 1e-14
        x = new
        if done:
            break
    phi, rho = x
    ll, b, s2 = conc([phi, rho])
    return RichResult(
        payload={
            "phi": phi,
            "rho": rho,
            "intercept": b[0],
            "beta": b[1:],
            "sigma2": s2,
            "sigma2_mu": phi * s2,
            "loglik": ll,
        }
    )


def sp_panel_dynamic(
    y, X, W, *, effects: str = "individual", space_time_lag: bool = True, bounds=(-0.99, 0.99)
) -> RichResult:
    r"""Dynamic spatial panel (spatial ARX): ``y_t = rho W y_t + gamma y_{t-1} + delta W y_{t-1} + X_t beta + mu + e_t``.

    Conditional on the first period, the lagged response ``y_{t-1}`` (and,
    when ``space_time_lag``, its spatial lag ``W y_{t-1}``) join ``X_t`` as
    regressors of the fixed-effects spatial lag model :func:`sp_panel_fe`
    (Elhorst 2014, ch. 4; the within estimator carries the Nickell bias of
    order ``1/T``, which Yu, de Jong and Lee 2008 correct for).
    ``beta`` lists ``gamma``, ``delta`` (if used), then the ``X`` coefficients.

    References
    ----------
    Elhorst, J. P. (2014). Spatial Econometrics, ch. 4. Yu, J., de Jong, R.
    and Lee, L.-F. (2008). Quasi-maximum likelihood estimators for spatial
    dynamic panel data with fixed effects when both n and T are large. J.
    Econometrics 146, 118-134.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> y = [[1.0, 2.0, 1.5], [2.0, 2.5, 1.0], [1.5, 3.0, 2.5], [2.5, 2.0, 3.0], [2.0, 1.0, 2.2]]
    >>> X = [[[0.5], [1.0], [0.2]], [[1.5], [0.7], [0.1]], [[0.4], [2.0], [1.1]], [[1.2], [0.3], [1.9]], [[0.3], [0.9], [1.4]]]
    >>> len(sp_panel_dynamic(y, X, W).beta)
    3
    """
    T, N = len(y), len(y[0])
    Wf = [[float(v) for v in row] for row in W]
    Xn = []
    for t in range(1, T):
        prev = [float(a) for a in y[t - 1]]
        wprev = _matvec(Wf, prev)
        rows = []
        for i in range(N):
            r = [prev[i]] + ([wprev[i]] if space_time_lag else []) + [float(v) for v in X[t][i]]
            rows.append(r)
        Xn.append(rows)
    return sp_panel_fe(y[1:], Xn, W, model="lag", effects=effects, bounds=bounds)


def cheatsheet() -> str:
    return "sp_panel_fe / sp_panel_re / sp_panel_dynamic -> spatial panel models (Elhorst 2014)."
