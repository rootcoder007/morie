# morie.fn -- function file (rootcoder007/morie)
"""Spatio-temporal building blocks from environmental epidemiology (Shaddick, Zidek and Schmidt
2023): Wishart and inverse-Wishart moments, the partitioned (Bartlett) decomposition of a
covariance matrix and Gaussian conditioning, separable prewhitening of a matrix-normal field,
the West-Harrison normal-gamma dynamic linear model, the Huerta et al. harmonic multi-site
ozone DLM, the Carroll et al. space-time correlation with its positive-definiteness check,
the Higdon process-convolution covariance and the theoretical ARMA autocorrelation function."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult
from ._sci_core import digamma

__all__ = [
    "wishart_moments",
    "partitioned_covariance",
    "separable_prewhiten",
    "normal_gamma_dlm",
    "harmonic_ozone_dlm",
    "carroll_st_correlation",
    "process_convolution_covariance",
    "arma_acf",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _mat(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in r] for r in a.tolist()]


def _mm(A, B):
    return [[ssum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _t(A):
    return [list(r) for r in zip(*A)]


def _logdet(A):
    n = len(A)
    M = [list(r) for r in A]
    ld = 0.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if p != c:
            M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        ld += math.log(abs(piv))
        for r in range(c + 1, n):
            f = M[r][c] / piv
            for k in range(c, n):
                M[r][k] -= f * M[c][k]
    return ld


def _eigh(A):
    w, v = np.linalg.eigh(np.asarray(A, dtype=float))
    return [float(x) for x in w.tolist()], [[float(x) for x in r] for r in v.tolist()]


def wishart_moments(scale, df, inverse_wishart=False):
    r"""Moments of the Wishart ``W_p(Sigma, delta)`` or inverse-Wishart ``W_p^{-1}(Psi, delta)``.

    Wishart: ``E(Z) = delta Sigma``, ``E(Z^{-1}) = Sigma^{-1} / (delta - p - 1)``
    (for ``delta > p + 1``) and ``E log|Z| = p log 2 + sum_{i=1}^p psi((delta - i + 1)/2)
    + log|Sigma|``. Inverse Wishart: ``E(Y) = Psi / (delta - p - 1)``,
    ``E(Y^{-1}) = delta Psi^{-1}`` and ``E log|Y| = -p log 2 - sum psi((delta - i + 1)/2)
    + log|Psi|`` (``psi`` the digamma function).

    References
    ----------
    Shaddick, G., Zidek, J. V. and Schmidt, A. M. (2023). *Spatio-Temporal
    Methods in Environmental Epidemiology with R*, 2nd edn. CRC Press,
    Appendix A.

    Examples
    --------
    >>> r = wishart_moments([[2.0, 0.5], [0.5, 1.0]], 6.0)
    >>> r.mean
    [[12.0, 3.0], [3.0, 6.0]]
    >>> round(r.mean_logdet, 12)
    3.571851124799
    """
    S = _mat(scale)
    p = len(S)
    d = float(df)
    if d <= p - 1:
        raise ValueError("df must exceed p - 1")
    Si = inverse(S)
    ld = _logdet(S)
    dig = ssum(digamma((d - i + 1) / 2.0) for i in range(1, p + 1))
    ok = d - p - 1 > 0
    if not inverse_wishart:
        mean = [[d * v for v in r] for r in S]
        mean_inv = [[v / (d - p - 1) for v in r] for r in Si] if ok else None
        mld = p * math.log(2.0) + dig + ld
    else:
        mean = [[v / (d - p - 1) for v in r] for r in S] if ok else None
        mean_inv = [[d * v for v in r] for r in Si]
        mld = -p * math.log(2.0) - dig + ld
    return RichResult(payload={"mean": mean, "mean_inverse": mean_inv, "mean_logdet": mld})


def partitioned_covariance(sigma, k, x2=None, mean=None):
    r"""Bartlett decomposition of a partitioned covariance and Gaussian conditioning.

    With ``Sigma`` split after its first ``k`` rows, ``Sigma_{1|2} = Sigma_11 -
    Sigma_12 Sigma_22^{-1} Sigma_21`` and ``tau = Sigma_12 Sigma_22^{-1}``;
    ``Sigma = T Delta T'`` with ``Delta = diag(Sigma_{1|2}, Sigma_22)`` and
    ``T = [[I, tau], [0, I]]``, i.e. ``Sigma = [[Sigma_{1|2} + tau Sigma_22 tau',
    tau Sigma_22], [Sigma_22 tau', Sigma_22]]``. Given ``x2`` (and ``mean``) the
    conditional mean of the first block is ``mu_1 + tau (x2 - mu_2)`` with
    covariance ``Sigma_{1|2}``.

    References
    ----------
    Shaddick, G., Zidek, J. V. and Schmidt, A. M. (2023). *Spatio-Temporal
    Methods in Environmental Epidemiology with R*, 2nd edn. CRC Press, eqs
    A.6-A.7.

    Examples
    --------
    >>> r = partitioned_covariance([[4.0, 2.0], [2.0, 2.0]], 1, x2=[1.0])
    >>> r.sigma_cond, r.tau, r.cond_mean
    ([[2.0]], [[1.0]], [1.0])
    """
    S = _mat(sigma)
    n = len(S)
    k = int(k)
    if not 0 < k < n:
        raise ValueError("k must split the matrix into two non-empty blocks")
    S11 = [r[:k] for r in S[:k]]
    S12 = [r[k:] for r in S[:k]]
    S22 = [r[k:] for r in S[k:]]
    tau = _mm(S12, inverse(S22))
    tS21 = _mm(tau, _t(S12))
    cond = [[S11[i][j] - tS21[i][j] for j in range(k)] for i in range(k)]
    T = [[float(i == j) for j in range(n)] for i in range(n)]
    for i in range(k):
        for j in range(n - k):
            T[i][k + j] = tau[i][j]
    D = [[0.0] * n for _ in range(n)]
    for i in range(k):
        for j in range(k):
            D[i][j] = cond[i][j]
    for i in range(n - k):
        for j in range(n - k):
            D[k + i][k + j] = S22[i][j]
    out = {"sigma_cond": cond, "tau": tau, "T": T, "Delta": D, "reconstructed": _mm(_mm(T, D), _t(T))}
    if x2 is not None:
        mu = [0.0] * n if mean is None else _vec(mean)
        dx = [v - m for v, m in zip(_vec(x2), mu[k:])]
        out["cond_mean"] = [mu[i] + ssum(tau[i][j] * dx[j] for j in range(n - k)) for i in range(k)]
    return RichResult(payload=out)


def separable_prewhiten(z, rho_t):
    r"""Temporal prewhitening of a separable space-time Gaussian field.

    For ``Z ~ N_{NS x NT}(mu, sigma^2 rho_S (x) rho_T)`` (matrix normal, sites
    by times), ``Z* = Z rho_T^{-1/2}`` has independent replicates over time:
    ``Z* ~ N(mu rho_T^{-1/2}, sigma^2 rho_S (x) I)``. ``rho_T^{-1/2}`` is the
    symmetric inverse square root from the eigen-decomposition.

    References
    ----------
    Shaddick, G., Zidek, J. V. and Schmidt, A. M. (2023). *Spatio-Temporal
    Methods in Environmental Epidemiology with R*, 2nd edn. CRC Press,
    section 12.3.

    Examples
    --------
    >>> r = separable_prewhiten([[1.0, 2.0]], [[1.0, 0.0], [0.0, 4.0]])
    >>> r.z_star
    [[1.0, 1.0]]
    """
    Z = _mat(z)
    R = _mat(rho_t)
    w, V = _eigh(R)
    if min(w) <= 0:
        raise ValueError("rho_t must be positive definite")
    n = len(R)
    Rm = [[ssum(V[i][m] * V[j][m] / math.sqrt(w[m]) for m in range(n)) for j in range(n)] for i in range(n)]
    return RichResult(payload={"z_star": _mm(Z, Rm), "rho_t_inv_sqrt": Rm})


def normal_gamma_dlm(y, F, G, W, m0, C0, n0, S0):
    r"""Normal dynamic linear model with unknown observation variance (normal-gamma filter).

    West and Harrison (1997, Theorem 4.3) in the scale-free form: with
    ``Z_{t-1} | V ~ N(m_{t-1}, V C*_{t-1})`` and ``1/V ~ Ga(n_{t-1}/2,
    n_{t-1} S_{t-1}/2)``: ``a_t = G m_{t-1}``, ``R*_t = G C*_{t-1} G' + W*``,
    ``f_t = F'a_t``, ``Q*_t = 1 + F'R*_t F``, ``A_t = R*_t F / Q*_t``,
    ``n_t = n_{t-1} + 1``, ``n_t S_t = n_{t-1} S_{t-1} + e_t^2 / Q*_t``,
    ``m_t = a_t + A_t e_t`` and ``C*_t = R*_t - A_t A_t' Q*_t``. The one-step
    forecast is Student t with ``n_{t-1}`` df, location ``f_t`` and scale
    ``S_{t-1} Q*_t``; ``loglik`` sums their log densities.

    References
    ----------
    West, M. and Harrison, J. (1997). *Bayesian Forecasting and Dynamic
    Models*, 2nd edn. Springer, section 4.6.

    Examples
    --------
    >>> r = normal_gamma_dlm([1.0, 1.5, 0.5], [1.0], [[1.0]], [[0.5]], [0.0], [[1.0]], 1.0, 1.0)
    >>> [round(v, 12) for v in r.S]
    [0.7, 0.595238095238, 0.486764705882]
    """
    yv = _vec(y)
    Fv = _vec(F)
    Gm, Wm = _mat(G), _mat(W)
    m, C = _vec(m0), _mat(C0)
    n, S = float(n0), float(S0)
    p = len(m)
    ms, Cs, fs, Qs, Ss, ns = [], [], [], [], [], []
    ll = 0.0
    for yt in yv:
        a = [ssum(Gm[i][j] * m[j] for j in range(p)) for i in range(p)]
        GC = _mm(Gm, C)
        R = [[ssum(GC[i][k] * Gm[j][k] for k in range(p)) + Wm[i][j] for j in range(p)] for i in range(p)]
        f = ssum(Fv[i] * a[i] for i in range(p))
        RF = [ssum(R[i][j] * Fv[j] for j in range(p)) for i in range(p)]
        Q = 1.0 + ssum(Fv[i] * RF[i] for i in range(p))
        e = yt - f
        sc = S * Q
        ll += (
            math.lgamma((n + 1) / 2)
            - math.lgamma(n / 2)
            - 0.5 * math.log(n * math.pi * sc)
            - (n + 1) / 2 * math.log(1 + e * e / (n * sc))
        )
        A = [v / Q for v in RF]
        nS = n * S + e * e / Q
        n += 1.0
        S = nS / n
        m = [a[i] + A[i] * e for i in range(p)]
        C = [[R[i][j] - A[i] * A[j] * Q for j in range(p)] for i in range(p)]
        ms.append(m)
        Cs.append([[S * v for v in r] for r in C])
        fs.append(f)
        Qs.append(Q)
        Ss.append(S)
        ns.append(n)
    return RichResult(payload={"m": ms, "C": Cs, "f": fs, "Q": Qs, "S": Ss, "n": ns, "loglik": ll})


def harmonic_ozone_dlm(y, times, coords, a, lam, sigma2, w, m0=None, c0=1e6):
    r"""Huerta et al. (2004) multi-site dynamic linear model for hourly ozone (Kalman filter).

    Data model ``Y_st = Z_1t + S_1t Z_2st + S_2t Z_3st + v_st`` with
    ``S_jt = cos(pi j t / 12) + a_j sin(pi j t / 12)`` (12- and 24-hour cycles)
    and ``Cov(v_t) = sigma2 exp(-D / lam)`` over the inter-site distances
    ``D``; process model: random walks ``Z_1t = Z_1(t-1) + gamma_1``,
    ``Z_jst = Z_js(t-1) + gamma_jst`` with variances ``w = (w1, w2, w3)``.
    ``y`` is times x sites; the state has ``1 + 2 NS`` elements with prior
    ``N(m0, c0 I)``. Returns filtered state means and the Gaussian
    log-likelihood of the one-step forecasts.

    References
    ----------
    Huerta, G., Sanso, B. and Stroud, J. R. (2004). A spatiotemporal model for
    Mexico City ozone levels. *Applied Statistics* 53, 231-248.

    Shaddick, G., Zidek, J. V. and Schmidt, A. M. (2023). *Spatio-Temporal
    Methods in Environmental Epidemiology with R*, 2nd edn. CRC Press,
    Example 12.8.

    Examples
    --------
    >>> r = harmonic_ozone_dlm([[1.0, 1.2], [0.8, 1.1]], [1, 2], [[0, 0], [1, 0]], [0.5, 0.2], 2.0, 0.1, [0.1, 0.05, 0.05])
    >>> round(r.loglik, 10)
    -29.6084570032
    """
    Y = _mat(y)
    tv = _vec(times)
    P = _mat(coords)
    ns = len(P)
    q = 1 + 2 * ns
    V = [
        [
            float(sigma2) * math.exp(-math.sqrt(ssum((P[i][d] - P[j][d]) ** 2 for d in range(len(P[0])))) / lam)
            for j in range(ns)
        ]
        for i in range(ns)
    ]
    Wd = [w[0]] + [w[1]] * ns + [w[2]] * ns
    m = [0.0] * q if m0 is None else _vec(m0)
    C = [[c0 if i == j else 0.0 for j in range(q)] for i in range(q)]
    means = []
    ll = 0.0
    for yt, t in zip(Y, tv):
        s1 = math.cos(math.pi * t / 12) + a[0] * math.sin(math.pi * t / 12)
        s2 = math.cos(math.pi * 2 * t / 12) + a[1] * math.sin(math.pi * 2 * t / 12)
        Fm = [[0.0] * q for _ in range(ns)]
        for s in range(ns):
            Fm[s][0] = 1.0
            Fm[s][1 + s] = s1
            Fm[s][1 + ns + s] = s2
        R = [[C[i][j] + (Wd[i] if i == j else 0.0) for j in range(q)] for i in range(q)]
        f = [ssum(Fm[s][i] * m[i] for i in range(q)) for s in range(ns)]
        FR = _mm(Fm, R)
        Qm = [[ssum(FR[s][i] * Fm[u][i] for i in range(q)) + V[s][u] for u in range(ns)] for s in range(ns)]
        e = [yt[s] - f[s] for s in range(ns)]
        Qi = inverse(Qm)
        Qe = [ssum(Qi[s][u] * e[u] for u in range(ns)) for s in range(ns)]
        ll += -0.5 * (ns * math.log(2 * math.pi) + _logdet(Qm) + ssum(e[s] * Qe[s] for s in range(ns)))
        K = _mm(_t(FR), Qi)
        m = [m[i] + ssum(K[i][s] * e[s] for s in range(ns)) for i in range(q)]
        C = [[R[i][j] - ssum(K[i][s] * FR[s][j] for s in range(ns)) for j in range(q)] for i in range(q)]
        means.append(m)
    return RichResult(payload={"means": means, "loglik": ll, "final_cov": C})


def carroll_st_correlation(coords, times, a, b):
    r"""Carroll et al. (1997) space-time correlation ``rho(d, v) = phi_v psi_v^d``.

    ``rho(0, 0) = 1``; otherwise ``phi_v psi_v^d`` with ``log psi_v = a0 + a1 v +
    a2 v^2`` and ``log phi_v = b0 + b1 v + b2 v^2`` (``d`` distance, ``v`` time
    lag). Cressie (1997) showed it is not positive definite in general; the
    smallest eigenvalue of the correlation matrix over the supplied
    space-time points is returned so the check can be made.

    References
    ----------
    Carroll, R. J., Chen, R., George, E. I., Li, T. H., Newton, H. J.,
    Schmiediche, H. and Wang, N. (1997). Ozone exposure and population density
    in Harris County, Texas. *JASA* 92, 392-404.

    Cressie, N. (1997). Comment. *JASA* 92, 411-413.

    Examples
    --------
    >>> r = carroll_st_correlation([[0, 0], [1, 0]], [0, 0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0])
    >>> [[round(v, 12) for v in row] for row in r.correlation]
    [[1.0, 0.606530659713], [0.606530659713, 1.0]]
    """
    P = _mat(coords)
    tv = _vec(times)
    n = len(P)
    Rm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d = math.sqrt(ssum((P[i][k] - P[j][k]) ** 2 for k in range(len(P[0]))))
            v = abs(tv[i] - tv[j])
            if d == 0 and v == 0:
                Rm[i][j] = 1.0
            else:
                psi = math.exp(a[0] + a[1] * v + a[2] * v * v)
                phi = math.exp(b[0] + b[1] * v + b[2] * v * v)
                Rm[i][j] = phi * psi**d
    w, _ = _eigh(Rm)
    return RichResult(payload={"correlation": Rm, "min_eigenvalue": min(w), "positive_definite": min(w) > 0})


def process_convolution_covariance(coords, kernels, sigma2=1.0, normalize=False):
    r"""Higdon process-convolution covariance with location-specific Gaussian kernels.

    With Gaussian kernels ``k_s = N(s, Sigma_s)`` convolved with white noise of
    variance ``sigma2``, ``C(s_i, s_j) = sigma2 (2 pi)^{-d/2} |Sigma_i + Sigma_j|^{-1/2}
    exp(-h' (Sigma_i + Sigma_j)^{-1} h / 2)``, ``h = s_i - s_j``. A common
    identity-type kernel gives the squared-exponential correlation of eq
    10.22. ``normalize`` returns the correlation ``C_ij / sqrt(C_ii C_jj)``
    (the Paciorek-Schervish nonstationary form). ``kernels`` is one ``d x d``
    matrix per location, or a single matrix for all.

    References
    ----------
    Higdon, D., Swall, J. and Kern, J. (1999). Non-stationary spatial modeling.
    *Bayesian Statistics* 6, 761-768.

    Paciorek, C. J. and Schervish, M. J. (2006). Spatial modelling using a new
    class of nonstationary covariance functions. *Environmetrics* 17, 483-506.

    Examples
    --------
    >>> r = process_convolution_covariance([[0.0, 0.0], [1.0, 0.0]], [[0.5, 0.0], [0.0, 0.5]], normalize=True)
    >>> round(r.covariance[0][1], 12)
    0.606530659713
    """
    P = _mat(coords)
    n, d = len(P), len(P[0])
    K = np.asarray(kernels, dtype=float)
    Ks = [_mat(k) for k in K.tolist()] if K.ndim == 3 else [_mat(kernels)] * n
    Cm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            S = [[Ks[i][r][c] + Ks[j][r][c] for c in range(d)] for r in range(d)]
            h = [P[i][c] - P[j][c] for c in range(d)]
            x = solve(S, h)
            q = ssum(h[c] * x[c] for c in range(d))
            Cm[i][j] = sigma2 * (2 * math.pi) ** (-d / 2) * math.exp(-0.5 * _logdet(S) - q / 2)
    if normalize:
        dg = [Cm[i][i] for i in range(n)]
        Cm = [[Cm[i][j] / math.sqrt(dg[i] * dg[j]) for j in range(n)] for i in range(n)]
    return RichResult(payload={"covariance": Cm})


def arma_acf(ar=(), ma=(), lag_max=None):
    r"""Theoretical autocorrelation function of a stationary ARMA(p, q) process.

    ``y_t = sum phi_j y_{t-j} + u_t + sum theta_j u_{t-j}``. The autocovariances
    ``gamma(0..r)``, ``r = max(p, q + 1)``, solve ``gamma(k) - sum_j phi_j
    gamma(|k - j|) = sum_{j=k}^q theta_j psi_{j-k}`` (``psi`` the MA(infinity)
    weights, ``theta_0 = 1``); further lags follow the AR recursion (the
    Yule-Walker equations ``gamma(tau) = sum phi_j gamma(tau - j)``). For
    MA(1), ``rho(1) = theta / (1 + theta^2)``. Agrees with ``stats::ARMAacf``.

    References
    ----------
    Brockwell, P. J. and Davis, R. A. (1991). *Time Series: Theory and
    Methods*, 2nd edn. Springer, section 3.3.

    Examples
    --------
    >>> arma_acf(ma=[0.5], lag_max=2)
    [1.0, 0.4, 0.0]
    >>> [round(v, 12) for v in arma_acf(ar=[0.5, 0.2], lag_max=3)]
    [1.0, 0.625, 0.5125, 0.38125]
    """
    phi, th = [float(v) for v in ar], [float(v) for v in ma]
    p, q = len(phi), len(th)
    if p == 0 and q == 0:
        raise ValueError("empty model")
    r = max(p, q + 1)
    lag_max = r if lag_max is None else int(lag_max)
    ph = phi + [0.0] * (r - p)
    theta = [1.0] + th
    psi = [1.0]
    for j in range(1, q + 1):
        psi.append(theta[j] + ssum(ph[i - 1] * psi[j - i] for i in range(1, min(j, r) + 1)))
    rhs = [ssum(theta[j] * psi[j - k] for j in range(k, q + 1)) if k <= q else 0.0 for k in range(r + 1)]
    A = [[0.0] * (r + 1) for _ in range(r + 1)]
    for k in range(r + 1):
        A[k][k] += 1.0
        for j in range(1, r + 1):
            A[k][abs(k - j)] -= ph[j - 1]
    g = solve(A, rhs)
    while len(g) <= lag_max:
        k = len(g)
        g.append(ssum(ph[j - 1] * g[k - j] for j in range(1, r + 1)))
    return [v / g[0] for v in g[: lag_max + 1]]


def cheatsheet() -> str:
    return (
        "wishart_moments / partitioned_covariance / separable_prewhiten / normal_gamma_dlm / "
        "harmonic_ozone_dlm / carroll_st_correlation / process_convolution_covariance / arma_acf."
    )
