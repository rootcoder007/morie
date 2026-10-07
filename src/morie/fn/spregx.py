# morie.fn -- function file (rootcoder007/morie)
"""Spatial regression extras: Moran's I permutation test, spautolm-type SAR/CAR error models with
weights, spatial two-stage least squares for the lag and Durbin models, the Kelejian-Prucha
heteroskedasticity-robust GM error model (KP-HET), the Kelejian-Piras spatial J-test and the
Satorra-Bentler scaled chi-square."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = [
    "moran_permutation_test",
    "spautolm_fit",
    "s2sls_lag",
    "gm_error_het",
    "spatial_j_test",
    "satorra_bentler",
]


def _mv(M, v):
    return [ssum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _cols(X):
    return [[float(r[k]) for r in X] for k in range(len(X[0]))] if X and len(X[0]) else []


def _rows(cols, n):
    return [[c[i] for c in cols] for i in range(n)]


def _mm(A, B):
    Bt = list(zip(*B))
    return [[ssum(a * b for a, b in zip(r, c)) for c in Bt] for r in A]


def _t(A):
    return [list(c) for c in zip(*A)]


def _pnorm(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def moran_permutation_test(x, W, *, nsim: int = 999, seed: int = 0) -> RichResult:
    r"""Moran's I with a permutation (Monte Carlo) test, as ``spdep::moran.mc``.

    ``I = (n / S0) z'Wz / z'z`` with ``z`` the centred values and ``S0`` the
    sum of the weights (``W`` as given, e.g. binary ``style = "B"``). The
    values are permuted ``nsim`` times (Fisher-Yates driven by Philox stream
    ``k`` of ``seed`` for permutation ``k``); the one-sided (greater)
    pseudo P-value is ``(1 + #{I_sim >= I}) / (nsim + 1)``.

    References
    ----------
    Moran, P. A. P. (1950). Notes on continuous stochastic phenomena.
    Biometrika 37, 17-23. Bivand, R. S., Pebesma, E. and Gomez-Rubio, V.
    (2013). Applied Spatial Data Analysis with R, 2nd ed., section 9.3.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> r = moran_permutation_test([1.0, 2.0, 3.0, 4.0], W, nsim=19)
    >>> round(r.statistic, 12)
    0.333333333333
    """
    xs = [float(v) for v in x]
    n = len(xs)
    Wf = [[float(v) for v in r] for r in W]
    s0 = ssum(v for r in Wf for v in r)

    def moran(v):
        m = ssum(v) / n
        z = [a - m for a in v]
        return n / s0 * ssum(z[i] * ssum(Wf[i][j] * z[j] for j in range(n)) for i in range(n)) / ssum(a * a for a in z)

    obs = moran(xs)
    sims = []
    for k in range(nsim):
        u = random_uniform(n, seed=seed, stream=k)
        p = list(xs)
        for i in range(n - 1, 0, -1):
            j = int(math.floor(float(u[i]) * (i + 1)))
            p[i], p[j] = p[j], p[i]
        sims.append(moran(p))
    ge = sum(1 for s in sims if s >= obs)
    return RichResult(payload={"statistic": obs, "p_value": (1 + ge) / (nsim + 1), "simulated": sims})


def _golden(f, lo, hi, tol=1e-9):
    r = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - r * (b - a), a + r * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - r * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + r * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def _bisect_score(g, x0, lo, hi, width=1e-5):
    a, b = max(lo, x0 - width), min(hi, x0 + width)
    fa, fb = g(a), g(b)
    if fa * fb > 0:
        return x0
    for _ in range(100):
        m = (a + b) / 2.0
        if m in (a, b):
            break
        fm = g(m)
        if fm == 0.0:
            return m
        if fa * fm < 0:
            b = m
        else:
            a, fa = m, fm
    return (a + b) / 2.0


def spautolm_fit(
    y, X, W, *, weights=None, family: str = "SAR", intercept: bool = True, bounds=(-0.99, 0.99)
) -> RichResult:
    r"""Spatial error models by maximum likelihood, as ``spatialreg::spautolm``.

    ``y = X beta + u`` with ``u`` simultaneous (SAR, ``(I - lambda W) u = e``)
    or conditional (CAR) autoregressive and ``Var(e_i) = sigma^2 / w_i``.
    With ``M = (I - lambda W)' D_w (I - lambda W)`` (SAR) or
    ``M = (I - lambda W) D_w`` (CAR), ``beta = (X'MX)^(-1) X'My``,
    ``SSE = e'Me`` and the profile log-likelihood
    ``J + (1/2) sum log w - (n/2) log(2 pi) - (n/2) log(SSE/n) - n/2``
    with ``J = ln|I - lambda W|`` (SAR) or half of it (CAR) is maximised over
    ``lambda`` (golden section, refined by bisection on its analytic slope).
    The weights reproduce ``spautolm(weights = POP8)`` in Bivand et al. (2013).

    References
    ----------
    Bivand, R. S., Pebesma, E. and Gomez-Rubio, V. (2013). Applied Spatial
    Data Analysis with R, 2nd ed., section 9.4. Waller, L. A. and Gotway, C.
    A. (2004). Applied Spatial Statistics for Public Health Data, ch. 9.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> r = spautolm_fit([1.0, 2.5, 2.0, 4.0], [[0.1], [0.5], [0.3], [0.9]], W)
    >>> round(r.lambda_, 6)
    -0.854441
    """
    ys = [float(v) for v in y]
    n = len(ys)
    Xr = [([1.0] if intercept else []) + [float(v) for v in r] for r in X]
    k = len(Xr[0])
    w = [1.0] * n if weights is None else [float(v) for v in weights]
    Wf = [[float(v) for v in r] for r in W]
    ev = [complex(v) for v in np.linalg.eigvals(np.asarray(Wf, dtype=float))]
    slw = ssum(math.log(v) for v in w)

    def mmat(lam):
        A = [[(1.0 if i == j else 0.0) - lam * Wf[i][j] for j in range(n)] for i in range(n)]
        if family == "CAR":
            return [[A[i][j] * w[j] for j in range(n)] for i in range(n)]
        DA = [[w[i] * A[i][j] for j in range(n)] for i in range(n)]
        return _mm(_t(A), DA)

    def fit(lam):
        M = mmat(lam)
        MX = _mm(M, Xr)
        My = _mv(M, ys)
        XMX = [[ssum(Xr[r][a] * MX[r][b] for r in range(n)) for b in range(k)] for a in range(k)]
        XMy = [ssum(Xr[r][a] * My[r] for r in range(n)) for a in range(k)]
        beta = solve(XMX, XMy)
        e = [ys[r] - ssum(Xr[r][j] * beta[j] for j in range(k)) for r in range(n)]
        sse = ssum(a * b for a, b in zip(e, _mv(M, e)))
        return beta, e, sse, XMX

    def ll(lam):
        sse = fit(lam)[2]
        ld = ssum(math.log(abs(1.0 - lam * v)) for v in ev)
        det = 0.5 * ld if family == "CAR" else ld
        return det + 0.5 * slw - n / 2 * math.log(2 * math.pi) - n / 2 * math.log(sse / n) - n / 2

    lam = _golden(lambda v: -ll(v), bounds[0], bounds[1])

    def slope(v):
        # envelope theorem: dSSE/dlambda = e'(dM/dlambda)e at the GLS beta
        e = fit(v)[1]
        We = _mv(Wf, e)
        if family == "CAR":
            dsse = -ssum(e[i] * We[i] * w[i] for i in range(n))
        else:
            Ae = [a - v * b for a, b in zip(e, We)]
            dsse = -2.0 * ssum(We[i] * w[i] * Ae[i] for i in range(n))
        dld = -ssum((u / (1.0 - v * u)).real for u in ev)
        return (0.5 * dld if family == "CAR" else dld) - n / 2 * dsse / fit(v)[2]

    lam = _bisect_score(slope, lam, bounds[0], bounds[1])
    beta, e, sse, XMX = fit(lam)
    s2 = sse / n
    V = inverse(XMX)
    return RichResult(
        payload={
            "lambda_": lam,
            "beta": beta,
            "sigma2": s2,
            "loglik": ll(lam),
            "se_beta": [math.sqrt(s2 * V[a][a]) for a in range(k)],
            "residuals": e,
            "family": family,
        }
    )


def _tsls(y, Z, H, het=False):
    n = len(y)
    HH = _mm(_t(H), H)
    HZ = _mm(_t(H), Z)
    bz = [solve(HH, [HZ[r][c] for r in range(len(HH))]) for c in range(len(Z[0]))]
    Zp = [[ssum(H[i][r] * bz[c][r] for r in range(len(HH))) for c in range(len(Z[0]))] for i in range(n)]
    ZpZp = _mm(_t(Zp), Zp)
    Zi = inverse(ZpZp)
    Zpy = [ssum(Zp[i][c] * y[i] for i in range(n)) for c in range(len(Z[0]))]
    delta = [ssum(Zi[a][b] * Zpy[b] for b in range(len(Zpy))) for a in range(len(Zpy))]
    e = [y[i] - ssum(Z[i][c] * delta[c] for c in range(len(delta))) for i in range(n)]
    df = n - len(delta)
    s2 = ssum(v * v for v in e) / df
    if het:
        ZoZ = [
            [ssum(Zp[i][a] * Zp[i][b] * e[i] * e[i] for i in range(n)) for b in range(len(delta))]
            for a in range(len(delta))
        ]
        V = _mm(_mm(Zi, ZoZ), Zi)
    else:
        V = [[v * s2 for v in row] for row in Zi]
    return delta, V, e, s2


def _lagcols(W, cols):
    return [_mv(W, c) for c in cols]


def s2sls_lag(y, X, W, *, durbin: bool = False, het: bool = False) -> RichResult:
    r"""Spatial two-stage least squares for the spatial lag (and spatial Durbin) model.

    ``y = rho W y + X beta (+ W X theta) + e``: ``W y`` is instrumented by
    ``H = [1, X, WX, W^2 X]`` (Kelejian and Prucha 1998), and for the
    Durbin model (``durbin=True``, Anselin's SDM two-step lag estimator) by
    ``H = [1, X, WX, W^2 X, W^3 X]``; ``delta = (Z_p'Z_p)^(-1) Z_p'y`` with
    ``Z_p`` the projection of ``Z = [1, X, (WX), Wy]`` on ``H``. Variances are
    ``s^2 (Z_p'Z_p)^(-1)`` (``s^2 = e'e/(n - k)``) or White-robust with
    ``het=True``; as ``sphet::spreg(model = "lag")``.

    References
    ----------
    Kelejian, H. H. and Prucha, I. R. (1998). A generalized spatial
    two-stage least squares procedure for estimating a spatial
    autoregressive model with autoregressive disturbances. J. Real Estate
    Finance Econ. 17, 99-121. Anselin, L. (1988). Spatial Econometrics:
    Methods and Models, ch. 7. Piras, G. (2010). sphet: spatial models with
    heteroskedastic innovations in R. J. Stat. Software 35(1).

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
    >>> r = s2sls_lag([1.0, 2.0, 1.5, 3.0, 2.5], [[0.1], [0.4], [0.35], [0.8], [0.2]], W)
    >>> len(r.coefficients)
    3
    """
    ys = [float(v) for v in y]
    n = len(ys)
    Xc = _cols(X)
    Wf = [[float(v) for v in r] for r in W]
    wx = _lagcols(Wf, Xc)
    wwx = _lagcols(Wf, wx)
    one = [1.0] * n
    if durbin:
        Z = [one] + Xc + wx
        H = [one] + Xc + wx + wwx + _lagcols(Wf, wwx)
    else:
        Z = [one] + Xc
        H = [one] + Xc + wx + wwx
    Z = Z + [_mv(Wf, ys)]
    delta, V, e, s2 = _tsls(ys, _rows(Z, n), _rows(H, n), het)
    return RichResult(
        payload={
            "coefficients": delta,
            "se": [math.sqrt(V[a][a]) for a in range(len(delta))],
            "rho": delta[-1],
            "residuals": e,
            "sigma2": s2,
            "vcov": V,
        }
    )


def _gg_het(W, u, n):
    ub = _mv(W, u)
    ubb = _mv(W, ub)
    # tr(W diag(d) W') = sum_i d_i sum_r W[r][i]^2
    cs = [ssum(W[r][i] ** 2 for r in range(n)) for i in range(n)]
    tra = ssum(ub[i] * u[i] * cs[i] for i in range(n))
    tt = ssum(ub[i] ** 2 * cs[i] for i in range(n))
    aa = ssum(u[i] ** 2 * cs[i] for i in range(n))
    first = ssum(a * b for a, b in zip(ubb, ub)) - tra
    second = ssum(a * a for a in ubb) - tt
    third = ssum(a * b for a, b in zip(u, ubb)) + ssum(a * a for a in ub)
    G = [[2 * first / n, -second / n], [third / n, -ssum(a * b for a, b in zip(ub, ubb)) / n]]
    g = [(ssum(a * a for a in ub) - aa) / n, ssum(a * b for a, b in zip(u, ub)) / n]
    return G, g


def _gm_obj(G, g, rho, P=None):
    v = [G[r][0] * rho + G[r][1] * rho * rho - g[r] for r in range(2)]
    if P is None:
        return v[0] ** 2 + v[1] ** 2
    return ssum(v[a] * P[a][b] * v[b] for a in range(2) for b in range(2))


def _gm_slope(G, g, rho, P=None):
    v = [G[r][0] * rho + G[r][1] * rho * rho - g[r] for r in range(2)]
    dv = [G[r][0] + 2 * G[r][1] * rho for r in range(2)]
    if P is None:
        return 2 * (v[0] * dv[0] + v[1] * dv[1])
    return 2 * ssum(dv[a] * P[a][b] * v[b] for a in range(2) for b in range(2))


def _gm_min(G, g, P=None, lo=-0.9 + 2.220446049250313e-16, hi=0.9 - 2.220446049250313e-16):
    r = _golden(lambda v: _gm_obj(G, g, v, P), lo, hi, 1e-10)
    return _bisect_score(lambda v: _gm_slope(G, g, v, P), r, lo, hi)


def gm_error_het(y, X, W) -> RichResult:
    r"""Heteroskedasticity-robust GM spatial error model (KP-HET), as ``sphet::spreg(model = "error", het = TRUE)``.

    ``y = X beta + u``, ``u = rho W u + e`` with ``E e_i^2`` unrestricted.
    Step 1: OLS residuals; ``rho`` by unweighted GM on the moments
    ``E[e'A_1 e] = 0``, ``E[e'A_2 e] = 0`` with ``A_1 = W'W - diag(W'W)`` and
    ``A_2 = W``. Step 2: two-stage least squares of ``y - rho W y`` on
    ``X - rho W X`` (instruments ``X``); ``rho`` re-estimated by efficient GMM
    with the heteroskedasticity-robust weight ``Phi^(-1)`` (traces
    ``tr((A_r + A_r') S (A_s + A_s') S) / 2`` plus ``a_r' S a_s``,
    ``S = diag(e^2)``), and the joint variance of ``(beta, rho)``.

    References
    ----------
    Kelejian, H. H. and Prucha, I. R. (2010). Specification and estimation
    of spatial autoregressive models with autoregressive and heteroskedastic
    disturbances. J. Econometrics 157, 53-67. Arraiz, I., Drukker, D. M.,
    Kelejian, H. H. and Prucha, I. R. (2010). A spatial Cliff-Ord-type model
    with heteroskedastic innovations. J. Regional Sci. 50, 592-614.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
    >>> r = gm_error_het([1.0, 2.0, 1.5, 3.0, 2.5], [[0.1], [0.4], [0.35], [0.8], [0.2]], W)
    >>> len(r.beta)
    2
    """
    ys = [float(v) for v in y]
    n = len(ys)
    Wf = [[float(v) for v in r] for r in W]
    H = [[1.0] + [float(v) for v in r] for r in X]
    k = len(H[0])
    Hc = _cols(H)
    b0, _, u0, _ = _tsls(ys, H, H)
    G, g = _gg_het(Wf, u0, n)
    rt = _gm_min(G, g)
    wy = _mv(Wf, ys)
    wZ = _lagcols(Wf, Hc)
    yt = [a - rt * b for a, b in zip(ys, wy)]
    Zt = _rows([[a - rt * b for a, b in zip(c, wc)] for c, wc in zip(Hc, wZ)], n)
    delta, _, _, _ = _tsls(yt, Zt, H)
    ut = [ys[i] - ssum(H[i][c] * delta[c] for c in range(k)) for i in range(n)]
    G, g = _gg_het(Wf, ut, n)
    A1 = _mm(_t(Wf), Wf)
    for i in range(n):
        A1[i][i] = 0.0
    S1 = [[A1[i][j] + A1[j][i] for j in range(n)] for i in range(n)]
    S2 = [[Wf[i][j] + Wf[j][i] for j in range(n)] for i in range(n)]
    HH = _mm(_t(H), H)
    HHi = inverse([[v / n for v in r] for r in HH])

    def psi(rho):
        Zs = _rows([[a - rho * b for a, b in zip(c, wc)] for c, wc in zip(Hc, wZ)], n)
        eps = [a - rho * b for a, b in zip(ut, _mv(Wf, ut))]
        Qhz = [[v / n for v in r] for r in _mm(_t(H), Zs)]
        sec = _mm(_mm(_t(Qhz), HHi), Qhz)
        Pm = _mm(_mm(HHi, Qhz), inverse(sec))
        Tm = _mm(H, Pm)
        # Z'(I - rho W)' (A + A') eps
        s1e, s2e = _mv(S1, eps), _mv(S2, eps)
        irw = [[(1.0 if i == j else 0.0) - rho * Wf[j][i] for j in range(n)] for i in range(n)]
        z1, z2 = _mv(irw, s1e), _mv(irw, s2e)
        al1 = [-ssum(H[i][c] * z1[i] for i in range(n)) / n for c in range(k)]
        al2 = [-ssum(H[i][c] * z2[i] for i in range(n)) / n for c in range(k)]
        a1, a2 = _mv(Tm, al1), _mv(Tm, al2)
        gam = [v * v for v in eps]

        def tr(A, B):
            return ssum(A[i][j] * gam[j] * B[j][i] * gam[i] for i in range(n) for j in range(n)) / 2

        p11 = (tr(S1, S1) + ssum(a1[i] * gam[i] * a1[i] for i in range(n))) / n
        p22 = (tr(S2, S2) + ssum(a2[i] * gam[i] * a2[i] for i in range(n))) / n
        p12 = (tr(S1, S2) + ssum(a1[i] * gam[i] * a2[i] for i in range(n))) / n
        Phi = [[p11, p12], [p12, p22]]
        return inverse(Phi), Pm, a1, a2, gam

    Pinv = psi(rt)[0]
    rf = _gm_min(G, g, Pinv)
    Pinv, Pm, a1, a2, gam = psi(rf)
    J = [G[0][0] + 2 * rf * G[0][1], G[1][0] + 2 * rf * G[1][1]]
    om_rr = 1.0 / ssum(J[a] * Pinv[a][b] * J[b] for a in range(2) for b in range(2))
    Psidd = [[ssum(H[i][a] * gam[i] * H[i][b] for i in range(n)) / n for b in range(k)] for a in range(k)]
    Psidr = [
        [ssum(H[i][a] * gam[i] * (a1[i] if c == 0 else a2[i]) for i in range(n)) / n for c in range(2)]
        for a in range(k)
    ]
    Omdd = _mm(_mm(_t(Pm), Psidd), Pm)
    PJ = [ssum(Pinv[a][b] * J[b] for b in range(2)) for a in range(2)]
    tmp = _mm(_t(Pm), Psidr)
    Omdr = [ssum(tmp[r][c] * PJ[c] for c in range(2)) * om_rr for r in range(k)]
    V = [[Omdd[a][b] / n for b in range(k)] + [Omdr[a] / n] for a in range(k)] + [[v / n for v in Omdr] + [om_rr / n]]
    return RichResult(
        payload={
            "beta": delta,
            "rho": rf,
            "rho_initial": rt,
            "vcov": V,
            "se": [math.sqrt(V[a][a]) for a in range(k + 1)],
            "residuals": ut,
        }
    )


def spatial_j_test(y, X0, W0, X1, W1, *, het: bool = False, method: str = "kelejian") -> RichResult:
    r"""Kelejian-Piras spatial J-test of a spatial lag model against a non-nested alternative.

    H1 (``X1``, ``W1``) is fitted by spatial 2SLS (:func:`s2sls_lag`) and its
    prediction ``yp = [1, X1, W1 y] delta_1`` is added to the H0 model
    ``y = [1, X0, W0 y, yp] gamma + e``, estimated by 2SLS with instruments
    ``[1, X0 & X1, W0 X0, W0^2 X0, W1 X1, W1^2 X1]`` (columns of ``X1`` that
    repeat ``X0`` are dropped). The statistic is the t ratio of ``yp``
    (normal reference); rejection favours the alternative. ``method="sphet"``
    reproduces ``sphet::kpjtest``, whose instrument set drops every column of
    ``X1`` when none of them repeats a column of ``X0`` (an R indexing quirk,
    ``x[, -integer(0)]``); the default keeps them, as Kelejian (2008).

    References
    ----------
    Kelejian, H. H. (2008). A spatial J-test for model specification against
    a single or a set of non-nested alternatives. Letters in Spatial and
    Resource Sciences 1, 3-11. Kelejian, H. H. and Piras, G. (2011). An
    extension of Kelejian's J-test for non-nested spatial models. Regional
    Sci. Urban Econ. 41, 281-292.

    Examples
    --------
    >>> W0 = [[0, 1, 0, 0, 0, 0], [0.5, 0, 0.5, 0, 0, 0], [0, 0.5, 0, 0.5, 0, 0], [0, 0, 0.5, 0, 0.5, 0], [0, 0, 0, 0.5, 0, 0.5], [0, 0, 0, 0, 1, 0]]
    >>> W1 = [[0, 0.5, 0.5, 0, 0, 0], [0.5, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 0, 0.5, 0], [0, 0.5, 0, 0, 0, 0.5], [0, 0, 0.5, 0, 0, 0.5], [0, 0, 0, 0.5, 0.5, 0]]
    >>> X = [[0.1, 1.0], [0.4, 0.2], [0.35, 0.9], [0.8, 0.4], [0.2, 0.7], [0.6, 0.3]]
    >>> r = spatial_j_test([1.0, 2.0, 1.5, 3.0, 2.5, 2.2], X, W0, X, W1)
    >>> 0.0 <= r.p_value <= 1.0
    True
    """
    ys = [float(v) for v in y]
    n = len(ys)
    W0f = [[float(v) for v in r] for r in W0]
    W1f = [[float(v) for v in r] for r in W1]
    alt = s2sls_lag(ys, X1, W1f)
    X0c, X1c = _cols(X0), _cols(X1)
    w1y = _mv(W1f, ys)
    yp = [ssum(v * c for v, c in zip([1.0] + [col[i] for col in X1c] + [w1y[i]], alt.coefficients)) for i in range(n)]
    extra = [c for c in X1c if all(c != d for d in X0c)]
    if method == "sphet" and len(extra) in (0, len(X1c)):
        extra = []
    w0x0 = _lagcols(W0f, X0c)
    w1x1 = _lagcols(W1f, X1c)
    H = [[1.0] * n] + X0c + extra + w0x0 + _lagcols(W0f, w0x0) + w1x1 + _lagcols(W1f, w1x1)
    Z = [[1.0] * n] + X0c + [_mv(W0f, ys), yp]
    delta, V, e, s2 = _tsls(ys, _rows(Z, n), _rows(H, n), het)
    t = delta[-1] / math.sqrt(V[-1][-1])
    return RichResult(
        payload={
            "statistic": t,
            "p_value": 2.0 * (1.0 - _pnorm(abs(t))),
            "coefficients": delta,
            "se": [math.sqrt(V[a][a]) for a in range(len(delta))],
        }
    )


def _vech_idx(p):
    return [(i, j) for j in range(p) for i in range(j, p)]


def satorra_bentler(data, sigma, delta, t_ml: float, df: int) -> RichResult:
    r"""Satorra-Bentler scaled chi-square ``T_SB = T_ML / c``.

    ``c = tr(U Gamma) / df`` with ``U = V - V Delta (Delta' V Delta)^(-1) Delta' V``,
    ``V = (1/2) D'(Sigma^(-1) x Sigma^(-1)) D`` (``D`` the duplication
    matrix, ``Sigma`` the model-implied covariance), ``Delta`` the Jacobian
    of ``vech(Sigma(theta))`` (rows in column-major ``vech`` order) and
    ``Gamma`` the asymptotic covariance of the sample covariances,
    ``(1/n) sum (d_i - dbar)(d_i - dbar)'`` with
    ``d_i = vech((x_i - xbar)(x_i - xbar)')`` (as lavaan's MLM).

    References
    ----------
    Satorra, A. and Bentler, P. M. (1994). Corrections to test statistics
    and standard errors in covariance structure analysis. In von Eye and
    Clogg (eds), Latent Variables Analysis, 399-419.

    Examples
    --------
    >>> X = [[1.0, 2.0], [2.0, 1.5], [3.0, 3.5], [4.0, 3.0], [2.5, 2.0]]
    >>> r = satorra_bentler(X, [[1.0, 0.5], [0.5, 1.0]], [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], 3.0, 1)
    >>> r.scaling > 0
    True
    """
    n, p = len(data), len(data[0])
    idx = _vech_idx(p)
    m = [ssum(r[j] for r in data) / n for j in range(p)]
    D = [[(r[i] - m[i]) * (r[j] - m[j]) for (i, j) in idx] for r in data]
    q = len(idx)
    dm = [ssum(row[a] for row in D) / n for a in range(q)]
    Gam = [[ssum((row[a] - dm[a]) * (row[b] - dm[b]) for row in D) / n for b in range(q)] for a in range(q)]
    Si = inverse([[float(v) for v in r] for r in sigma])
    # V[(i,j),(k,l)] = (1/2) D'(Si x Si) D entries: c_ij c_kl (Si_ik Si_jl + Si_il Si_jk) / 2 with
    # c = 1 on the diagonal of Sigma, 2 off it, halved again for the symmetric sum
    V = [[0.0] * q for _ in range(q)]
    for a, (i, j) in enumerate(idx):
        for b, (k, m) in enumerate(idx):
            v = 0.5 * (Si[i][k] * Si[j][m] + Si[i][m] * Si[j][k])
            ca = 1.0 if i == j else 2.0
            cb = 1.0 if k == m else 2.0
            V[a][b] = 0.5 * ca * cb * v
    Dl = [[float(v) for v in r] for r in delta]
    VD = _mm(V, Dl)
    M = inverse(_mm(_t(Dl), VD))
    U = [
        [
            V[a][b] - ssum(VD[a][r] * ssum(M[r][s] * VD[b][s] for s in range(len(M))) for r in range(len(M)))
            for b in range(q)
        ]
        for a in range(q)
    ]
    trUG = ssum(U[a][b] * Gam[b][a] for a in range(q) for b in range(q))
    c = trUG / df
    return RichResult(payload={"statistic": t_ml / c, "scaling": c, "gamma": Gam})


def cheatsheet() -> str:
    return (
        "moran_permutation_test / spautolm_fit / s2sls_lag / gm_error_het / spatial_j_test / satorra_bentler -> "
        "spatial regression extras and the Satorra-Bentler scaled chi-square."
    )


# alias kept from the retired placeholder of the same name
sem_sb_chi_sq = satorra_bentler
