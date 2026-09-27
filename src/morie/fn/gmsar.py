# morie.fn -- function file (rootcoder007/morie)
"""Generalised moments estimators: spatial error model and SAC by GS2SLS (Kelejian and Prucha)."""

from __future__ import annotations

from ._containers import DescriptiveResult
from ._qpcore import inverse, ssum
from .sarreg import _brent, _mv, _ols

__all__ = ["gm_error_sar", "gs2sls_sac", "kp_moments"]


def _lists(A):
    return [[float(v) for v in r] for r in A]


def _t(A):
    return [list(c) for c in zip(*A)]


def _mm(A, B):
    Bt = _t(B)
    return [[float(ssum(a * b for a, b in zip(r, c))) for c in Bt] for r in A]


def _dot(a, b):
    return ssum(x * y for x, y in zip(a, b))


def kp_moments(u, W):
    """Moment matrix ``G`` (3 x 3) and vector ``g`` of Kelejian and Prucha (1999).

    With ``u`` the residuals, ``wu = Wu`` and ``wwu = W Wu``, the three
    moment conditions are ``G (lambda, lambda^2, sigma^2)' = g`` where
    ``G`` has columns ``(2u'wu, 2wwu'wu, u'wwu + wu'wu) / n``,
    ``-(wu'wu, wwu'wwu, wwu'wu) / n`` and ``(1, tr(W'W) / n, 0)`` and
    ``g = (u'u, wu'wu, u'wu) / n`` (``spatialreg:::.kpwuwu``).
    """
    n = len(u)
    wu = _mv(W, u)
    wwu = _mv(W, wu)
    trwpw = ssum(v * v for r in W for v in r)
    G = [
        [2 * _dot(u, wu) / n, -_dot(wu, wu) / n, 1.0],
        [2 * _dot(wwu, wu) / n, -_dot(wwu, wwu) / n, trwpw / n],
        [(_dot(u, wwu) + _dot(wu, wu)) / n, -_dot(wwu, wu) / n, 0.0],
    ]
    g = [_dot(u, u) / n, _dot(wu, wu) / n, _dot(u, wu) / n]
    return G, g, trwpw, wu, wwu


def _gm_solve(G, g, interval):
    c3 = [r[2] for r in G]
    c33 = _dot(c3, c3)

    def prof(lam):
        r = [g[i] - G[i][0] * lam - G[i][1] * lam * lam for i in range(3)]
        return _dot(r, r) - _dot(c3, r) ** 2 / c33

    lo, hi = interval
    grid = [lo + (hi - lo) * k / 400 for k in range(401)]
    k = min(range(401), key=lambda i: prof(grid[i]))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, 400)]
    lam, _f = _brent(prof, a, b)
    # the profile is a quartic in lambda: polish the root of its derivative by Newton
    G1 = [G[i][0] for i in range(3)]
    G2 = [G[i][1] for i in range(3)]
    for _ in range(20):
        r = [g[i] - G1[i] * lam - G2[i] * lam * lam for i in range(3)]
        d1 = [-G1[i] - 2 * G2[i] * lam for i in range(3)]
        d2 = [-2 * G2[i] for i in range(3)]
        cr, cd1, cd2 = _dot(c3, r), _dot(c3, d1), _dot(c3, d2)
        f1 = 2 * _dot(r, d1) - 2 * cr * cd1 / c33
        f2 = 2 * (_dot(d1, d1) + _dot(r, d2)) - 2 * (cd1 * cd1 + cr * cd2) / c33
        if f2 <= 0.0:
            break
        step = f1 / f2
        lam -= step
        if abs(step) <= 1e-15 * max(1.0, abs(lam)):
            break
    r = [g[i] - G[i][0] * lam - G[i][1] * lam * lam for i in range(3)]
    return lam, _dot(c3, r) / c33


def _tsls(y, yend, X, Q, robust, sig2n_k):
    n = len(y)
    Qc = _t(Q)
    QQi = _lists(inverse([[_dot(a, b) for b in Qc] for a in Qc]))
    bz = [ssum(QQi[i][j] * _dot(Qc[j], yend) for j in range(len(Qc))) for i in range(len(Qc))]
    yendp = [_dot(r, bz) for r in Q]
    Zp = [[yendp[i]] + list(X[i]) for i in range(n)]
    Z = [[yend[i]] + list(X[i]) for i in range(n)]
    Zc = _t(Zp)
    ZZi = _lists(inverse([[_dot(a, b) for b in Zc] for a in Zc]))
    Zy = [_dot(c, y) for c in Zc]
    biv = [_dot(r, Zy) for r in ZZi]
    e = [y[i] - _dot(Z[i], biv) for i in range(n)]
    sse = _dot(e, e)
    p = len(Zc)
    df = n - p if sig2n_k else n
    if robust is None:
        var = [[v * sse / df for v in r] for r in ZZi]
    else:
        f = n / df if robust == "HC1" else 1.0
        om = [f * v * v for v in e]
        M = [[ssum(Zc[a][i] * Zc[b][i] * om[i] for i in range(n)) for b in range(p)] for a in range(p)]
        var = _mm(_mm(ZZi, M), ZZi)
    return biv, var, sse, e, df


def gm_error_sar(
    y, X, W, *, se_lambda: bool = True, lambda_se_method: str = "trace", interval=(-0.999, 0.999)
) -> DescriptiveResult:
    r"""Spatial error model by generalised moments (Kelejian and Prucha 1999).

    Model ``y = X beta + u``, ``u = lambda W u + e``.  The OLS residuals
    give the moment system of :func:`kp_moments`; ``(lambda, sigma^2)``
    minimise ``|G (lambda, lambda^2, sigma^2)' - g|^2`` (profiled over
    ``sigma^2``, then Brent on ``lambda`` in ``interval``), and ``beta`` is
    feasible GLS: OLS of ``y - lambda Wy`` on ``X - lambda WX``.  As
    ``spatialreg::GMerrorsar`` (non-legacy), the coefficient covariance is
    ``s^2 (B'B)^{-1}`` with ``B = X - lambda WX`` and
    ``s^2 = |e - lambda W e|^2 / n`` from the OLS residuals ``e``.

    The standard error of ``lambda`` follows Kelejian and Prucha (2004):
    ``omega = (J_1'J_1)^{-1} J_1' Phi J_1 (J_1'J_1)^{-1}``,
    ``se = sqrt(omega / n)``, with
    ``Phi_rs = s^4 tr[(A_r + A_r')(A_s + A_s')] / (2n)``,
    ``A_2 = W'W`` and ``A_1 = c (W'W - tr(W'W)/n I)``,
    ``c = 1 / sqrt(1 + (tr(W'W)/n)^2)``.  ``spatialreg`` takes the trace
    for ``Phi_11`` but sums every entry of the matrix product for
    ``Phi_12`` and ``Phi_22``.  ``lambda_se_method="trace"`` (default)
    uses the trace throughout, as Kelejian and Prucha (2004);
    ``"spatialreg"`` reproduces ``spatialreg``'s entry sums.

    :param y: Response (n,).
    :param X: Design matrix (n, p), including any intercept column.
    :param W: Spatial weights (n, n).
    :param se_lambda: Also compute the standard error of ``lambda``.
    :param lambda_se_method: ``"trace"`` or ``"spatialreg"`` (see above).
    :param interval: Search interval for ``lambda``.
    :return: :class:`DescriptiveResult` with ``lambda``, ``sigma2_gm``,
        ``coefficients``, ``se``, ``s2``, ``lambda_se``, ``fitted``,
        ``residuals``.

    References
    ----------
    Kelejian, H. H. and Prucha, I. R. (1999). A generalized moments
    estimator for the autoregressive parameter in a spatial model.
    *International Economic Review*, 40(2), 509-533.
    Kelejian, H. H. and Prucha, I. R. (2004). Estimation of simultaneous
    systems of spatially interrelated cross sectional equations. *Journal
    of Econometrics*, 118(1-2), 27-50.

    Examples
    --------
    >>> n = 8
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (0.1, 0.6, 0.2, 0.9, 0.3, 0.5, 0.8, 0.4)]
    >>> r = gm_error_sar([1.0, 2.2, 1.4, 3.1, 0.9, 2.0, 2.6, 1.1], X, W)
    >>> round(r.value["lambda"], 6)
    -0.042117
    """
    y = [float(v) for v in y]
    X = [[float(v) for v in r] for r in X]
    W = [[float(v) for v in r] for r in W]
    n, p = len(y), len(X[0])
    b0 = [float(v) for v in _ols(X, y)]
    e = [y[i] - _dot(X[i], b0) for i in range(n)]
    G, g, trwpw, wu, wwu = kp_moments(e, W)
    lam, s2gm = _gm_solve(G, g, interval)
    Wy = _mv(W, y)
    cols = _t(X)
    WX = _t([_mv(W, c) for c in cols])
    B = [[X[i][k] - lam * WX[i][k] for k in range(p)] for i in range(n)]
    beta = [float(v) for v in _ols(B, [y[i] - lam * Wy[i] for i in range(n)])]
    fit = [_dot(r, beta) for r in X]
    et = [e[i] - lam * wu[i] for i in range(n)]
    s2 = _dot(et, et) / n
    Bc = _t(B)
    cov = [[v * s2 for v in r] for r in _lists(inverse([[_dot(a, b) for b in Bc] for a in Bc]))]
    lse = None
    if se_lambda:
        a = trwpw / n
        c = (1.0 / (1.0 + a * a)) ** 0.5
        J = [
            [2 * c * (_dot(wwu, wu) - a * _dot(wu, e)), -c * (_dot(wwu, wwu) - a * _dot(wu, wu))],
            [_dot(wwu, e) + _dot(wu, wu), -_dot(wwu, wu)],
        ]
        J1 = [(J[r][0] + J[r][1] * 2 * lam) / n for r in range(2)]
        A2 = _mm(_t(W), W)
        A2s = [[A2[i][j] + A2[j][i] for j in range(n)] for i in range(n)]
        A1s = [[c * (A2s[i][j] - (2 * a if i == j else 0.0)) for j in range(n)] for i in range(n)]

        def tr(P, Q):
            return ssum(P[i][j] * Q[j][i] for i in range(n) for j in range(n))

        def allsum(P, Q):
            # sum of every entry of P'Q (spatialreg): sum_k rowsum_k(P) rowsum_k(Q)
            return ssum(ssum(P[k]) * ssum(Q[k]) for k in range(n))

        if lambda_se_method not in ("trace", "spatialreg"):
            raise ValueError("lambda_se_method must be 'trace' or 'spatialreg'")
        f12 = tr if lambda_se_method == "trace" else allsum
        s4 = s2 * s2
        p12 = f12(A1s, A2s)
        phi = [[s4 * tr(A1s, A1s) / (2 * n), s4 * p12 / (2 * n)], [s4 * p12 / (2 * n), s4 * f12(A2s, A2s) / (2 * n)]]
        jj = _dot(J1, J1)
        om = ssum(J1[r] * phi[r][s] * J1[s] for r in range(2) for s in range(2)) / (jj * jj)
        lse = (om / n) ** 0.5
    return DescriptiveResult(
        name="gm_error_sar",
        value={
            "lambda": lam,
            "sigma2_gm": s2gm,
            "coefficients": beta,
            "se": [cov[k][k] ** 0.5 for k in range(p)],
            "s2": s2,
            "lambda_se": lse,
            "fitted": fit,
            "residuals": [y[i] - fit[i] for i in range(n)],
        },
    )


def gs2sls_sac(y, X, W, W2=None, *, robust=None, sig2n_k: bool = False, interval=(-0.999, 0.999)) -> DescriptiveResult:
    r"""SAC (SARAR) model by generalised spatial two-stage least squares.

    Model ``y = rho W y + X beta + u``, ``u = lambda W_2 u + e``
    (Kelejian and Prucha 1998), as ``spatialreg::gstsls``:

    1. 2SLS of ``y`` on ``(Wy, X)`` with instruments ``X, WX, W^2X`` (lags
       of the non-intercept columns);
    2. ``lambda`` from the generalised moments of the step-1 residuals on
       ``W_2`` (:func:`kp_moments`);
    3. 2SLS of ``y - lambda W_2 y`` on ``(Wy - lambda W_2 Wy,
       X - lambda W_2 X)`` with the same instruments ``WX, W^2X`` and the
       transformed ``X``.

    The covariance of ``(rho, beta)`` is ``s^2 (Z_p'Z_p)^{-1}`` with
    ``s^2 = e'e / df`` (``df = n`` unless ``sig2n_k``), or the HC0 / HC1
    sandwich when ``robust`` is given.  Residuals are those of step 3.

    :param y: Response (n,).
    :param X: Design matrix (n, p); a leading column of ones is the intercept.
    :param W: Weights of the spatial lag.
    :param W2: Weights of the error process (default ``W``).
    :param robust: ``None``, ``"HC0"`` or ``"HC1"``.
    :param sig2n_k: Divide the residual sum of squares by ``n - k``.
    :param interval: Search interval for ``lambda``.
    :return: :class:`DescriptiveResult` with ``rho``, ``lambda``,
        ``sigma2_gm``, ``coefficients`` (``rho`` first), ``se``, ``cov``,
        ``s2``, ``residuals``.

    References
    ----------
    Kelejian, H. H. and Prucha, I. R. (1998). A generalized spatial
    two-stage least squares procedure for estimating a spatial
    autoregressive model with autoregressive disturbances. *Journal of
    Real Estate Finance and Economics*, 17(1), 99-121.

    Examples
    --------
    >>> n = 8
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (0.1, 0.6, 0.2, 0.9, 0.3, 0.5, 0.8, 0.4)]
    >>> r = gs2sls_sac([1.0, 2.2, 1.4, 3.1, 0.9, 2.0, 2.6, 1.1], X, W)
    >>> round(r.value["rho"], 6)
    -0.561259
    """
    y = [float(v) for v in y]
    X = [[float(v) for v in r] for r in X]
    W = [[float(v) for v in r] for r in W]
    W2 = W if W2 is None else [[float(v) for v in r] for r in W2]
    n, p = len(y), len(X[0])
    cols = _t(X)
    start = 1 if all(v == 1.0 for v in cols[0]) else 0
    if start >= p:
        raise ValueError("X needs a non-intercept column to build instruments")
    WXc = [_mv(W, cols[k]) for k in range(start, p)]
    WWXc = [_mv(W, c) for c in WXc]
    inst = _t(WXc + WWXc)
    Wy = _mv(W, y)
    Q = [list(X[i]) + inst[i] for i in range(n)]
    _b, _v, _s, u, _d = _tsls(y, Wy, X, Q, robust, sig2n_k)
    G, g, *_rest = kp_moments(u, W2)
    lam, s2gm = _gm_solve(G, g, interval)
    W2y = _mv(W2, y)
    W2Wy = _mv(W2, Wy)
    W2X = _t([_mv(W2, c) for c in cols])
    yt = [y[i] - lam * W2y[i] for i in range(n)]
    xt = [[X[i][k] - lam * W2X[i][k] for k in range(p)] for i in range(n)]
    wyt = [Wy[i] - lam * W2Wy[i] for i in range(n)]
    Qt = [xt[i] + inst[i] for i in range(n)]
    coef, var, sse, e, df = _tsls(yt, wyt, xt, Qt, robust, sig2n_k)
    return DescriptiveResult(
        name="gs2sls_sac",
        value={
            "rho": coef[0],
            "lambda": lam,
            "sigma2_gm": s2gm,
            "coefficients": coef,
            "se": [var[k][k] ** 0.5 for k in range(len(coef))],
            "cov": var,
            "s2": sse / df,
            "residuals": e,
        },
    )


def cheatsheet() -> str:
    return "gm_error_sar / gs2sls_sac -> Kelejian-Prucha GM error model and GS2SLS SAC (spatialreg GMerrorsar, gstsls)."
