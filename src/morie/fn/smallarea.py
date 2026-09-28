# morie.fn -- function file (rootcoder007/morie)
"""Small-area and disease-rate estimation: the Fay-Herriot area-level EBLUP with its MSE, the
Battese-Harter-Fuller unit-level EBLUP, Marshall's empirical-Bayes rates and the
Potthoff-Whittinghill test of rate homogeneity (Poisson-gamma empirical Bayes is
:func:`morie.fn.dismap.poisson_gamma_eb`)."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rrng_core import pnorm

__all__ = ["fay_herriot", "bhf_eblup", "marshall_eb", "potthoff_whittinghill"]


def _mat(X):
    return [[float(v) for v in r] for r in (X.tolist() if hasattr(X, "tolist") else X)]


def _inv(A):
    return [[float(v) for v in r] for r in inverse([list(r) for r in A])]


def _logdet(A):
    """log|A| of a symmetric positive-definite matrix (Cholesky)."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(s) if i == j else s / L[j][j]
    return 2.0 * ssum(math.log(L[i][i]) for i in range(n))


def fay_herriot(y, X, vardir, *, method: str = "REML", maxiter: int = 100, tol: float = 1e-10) -> RichResult:
    r"""Fay-Herriot area-level EBLUP ``theta_d = x_d'beta + gamma_d (y_d - x_d'beta)``, ``gamma_d = A / (A + D_d)``, with its MSE (as ``sae::eblupFH`` / ``mseFH``).

    Model ``y_d = x_d'beta + v_d + e_d``, ``v_d ~ N(0, A)``, ``e_d ~ N(0,
    D_d)`` with known sampling variances ``D_d`` (``vardir``). ``A`` by Fisher
    scoring from ``median(D)``: ``REML``, ``ML`` or the ``FH`` moment
    equation ``sum (y - X beta)^2 / (A + D) = m - p``; truncated at 0.
    Iteration stops when ``|A_{k+1} - A_k| / |A_k| < tol`` (sae's
    ``PRECISION``). MSE = ``g1 + g2 + 2 g3`` (Prasad-Rao for REML; ML adds the
    Datta-Lahiri bias term, FH the Datta-Rao-Smith one).

    :return: ``A``, ``beta``, ``se_beta``, ``eblup``, ``gamma``, ``mse``,
        ``loglik``, ``aic``, ``bic``, ``iterations``, ``converged``.

    References
    ----------
    Fay, R. E. and Herriot, R. A. (1979). Estimates of income for small places:
    an application of James-Stein procedures to census data. *JASA*, 74(366),
    269-277.
    Datta, G. S., Rao, J. N. K. and Smith, D. D. (2005). On measuring the
    variability of small area estimators under a basic area level model.
    *Biometrika*, 92(1), 183-196.
    Rao, J. N. K. and Molina, I. (2015). *Small Area Estimation* (2nd ed.).
    Wiley.

    Examples
    --------
    >>> r = fay_herriot([1.0, 2.0, 3.0, 5.0], [[1], [1], [1], [1]], [1.0, 1.0, 1.0, 1.0])
    >>> round(r.A, 10), round(r.beta[0], 10)
    (1.9166666667, 2.75)
    """
    if method not in ("REML", "ML", "FH"):
        raise ValueError("method must be REML, ML or FH")
    yv = [float(v) for v in y]
    Xm = _mat(X)
    D = [float(v) for v in vardir]
    m, p = len(yv), len(Xm[0])

    def gls(A):
        Vi = [1.0 / (A + d) for d in D]
        Q = _inv([[ssum(Vi[i] * Xm[i][a] * Xm[i][b] for i in range(m)) for b in range(p)] for a in range(p)])
        Xty = [ssum(Vi[i] * Xm[i][a] * yv[i] for i in range(m)) for a in range(p)]
        beta = [ssum(Q[a][b] * Xty[b] for b in range(p)) for a in range(p)]
        res = [yv[i] - ssum(Xm[i][a] * beta[a] for a in range(p)) for i in range(m)]
        return Vi, Q, beta, res

    A = sorted(D)[m // 2] if m % 2 else 0.5 * (sorted(D)[m // 2 - 1] + sorted(D)[m // 2])
    k, diff = 0, tol + 1.0
    while diff > tol and k < maxiter:
        k += 1
        Vi, Q, beta, res = gls(A)
        if method == "FH":
            s = ssum(r * r * v for r, v in zip(res, Vi)) - (m - p)
            F = ssum(Vi)
        else:
            # P = V^-1 - V^-1 X Q X' V^-1; P y = V^-1 res
            Py = [v * r for v, r in zip(Vi, res)]
            if method == "ML":
                s = -0.5 * ssum(Vi) + 0.5 * ssum(v * v for v in Py)
                F = 0.5 * ssum(v * v for v in Vi)
            else:
                XV = [[Vi[i] * Xm[i][a] for a in range(p)] for i in range(m)]
                H = [[ssum(XV[i][a] * Q[a][b] for a in range(p)) for b in range(p)] for i in range(m)]
                P = [
                    [(Vi[i] if i == j else 0.0) - ssum(H[i][b] * XV[j][b] for b in range(p)) for j in range(m)]
                    for i in range(m)
                ]
                s = -0.5 * ssum(P[i][i] for i in range(m)) + 0.5 * ssum(v * v for v in Py)
                F = 0.5 * ssum(P[i][j] * P[j][i] for i in range(m) for j in range(m))
        An = A + s / F
        diff = abs((An - A) / A) if A != 0 else abs(An - A)
        A = An
    converged = diff <= tol
    A = max(A, 0.0)
    Vi, Q, beta, res = gls(A)
    se = [math.sqrt(Q[a][a]) for a in range(p)]
    gam = [A * v for v in Vi]
    eblup = [yv[i] - res[i] + gam[i] * res[i] for i in range(m)]
    ll = -0.5 * ssum(math.log(2 * math.pi * (A + d)) + r * r / (A + d) for d, r in zip(D, res))
    B = [d * v for d, v in zip(D, Vi)]
    S2 = ssum(v * v for v in Vi)
    if method == "FH":
        S1 = ssum(Vi)
        varA, bias = 2.0 * m / S1**2, 2.0 * (m * S2 - S1**2) / S1**3
    else:
        varA = 2.0 / S2
        if method == "ML":
            M = [[ssum(Vi[i] ** 2 * Xm[i][a] * Xm[i][b] for i in range(m)) for b in range(p)] for a in range(p)]
            bias = -ssum(Q[a][b] * M[b][a] for a in range(p) for b in range(p)) / S2
        else:
            bias = 0.0
    mse = []
    for i in range(m):
        g1 = D[i] * (1.0 - B[i])
        g2 = B[i] ** 2 * ssum(Xm[i][a] * Q[a][b] * Xm[i][b] for a in range(p) for b in range(p))
        g3 = B[i] ** 2 * varA / (A + D[i])
        mse.append(g1 + g2 + 2.0 * g3 - bias * B[i] ** 2)
    return RichResult(
        payload={
            "A": A,
            "beta": beta,
            "se_beta": se,
            "eblup": eblup,
            "gamma": gam,
            "mse": mse,
            "loglik": ll,
            "aic": -2 * ll + 2 * (p + 1),
            "bic": -2 * ll + (p + 1) * math.log(m),
            "iterations": k,
            "converged": converged,
            "method": method,
        }
    )


def _bhf_profile(rho, groups, p, reml):
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for Xg, yg in groups:
        c = rho / (1.0 + len(yg) * rho)
        sx = [ssum(r[a] for r in Xg) for a in range(p)]
        sy = ssum(yg)
        for a in range(p):
            Xty[a] += ssum(r[a] * v for r, v in zip(Xg, yg)) - c * sx[a] * sy
            for b in range(p):
                XtX[a][b] += ssum(r[a] * r[b] for r in Xg) - c * sx[a] * sx[b]
    Q = _inv(XtX)
    beta = [ssum(Q[a][b] * Xty[b] for b in range(p)) for a in range(p)]
    rss = 0.0
    N = 0
    for Xg, yg in groups:
        c = rho / (1.0 + len(yg) * rho)
        r = [v - ssum(x[a] * beta[a] for a in range(p)) for x, v in zip(Xg, yg)]
        rss += ssum(v * v for v in r) - c * ssum(r) ** 2
        N += len(yg)
    ld = ssum(math.log1p(len(yg) * rho) for _, yg in groups)
    dof = N - p if reml else N
    s2 = rss / dof
    ll = -0.5 * (dof * (math.log(2 * math.pi * s2) + 1.0) + ld + (_logdet(XtX) if reml else 0.0))
    # d ll / d rho: dH^-1/drho = -J / (1 + n rho)^2 blockwise; beta is the GLS minimiser (envelope)
    drss = dld = dq = 0.0
    for Xg, yg in groups:
        q = (1.0 + len(yg) * rho) ** 2
        r = [v - ssum(x[a] * beta[a] for a in range(p)) for x, v in zip(Xg, yg)]
        sx = [ssum(x[a] for x in Xg) for a in range(p)]
        drss -= ssum(r) ** 2 / q
        dld += len(yg) / (1.0 + len(yg) * rho)
        dq -= ssum(sx[a] * Q[a][b] * sx[b] for a in range(p) for b in range(p)) / q
    score = -0.5 * (dof * drss / rss + dld + (dq if reml else 0.0))
    return ll, beta, s2, Q, score


def bhf_eblup(y, X, area, xbar_pop, *, areas=None, popsize=None, method: str = "REML") -> RichResult:
    r"""Battese-Harter-Fuller unit-level EBLUP of area means under the nested-error model ``y_dj = x_dj'beta + u_d + e_dj`` (as ``sae::eblupBHF``).

    Variance components by REML or ML, profiling ``sigma_e^2`` out and
    solving the analytic profile score in ``theta = log(sigma_u^2 /
    sigma_e^2)`` by bisection on ``[-30, 15]`` (a non-positive score at the
    lower end gives the boundary ``sigma_u^2 = 0``; the profile is assumed
    unimodal). With ``gamma_d = sigma_u^2 / (sigma_u^2 + sigma_e^2 /
    n_d)``, ``u_d = gamma_d (ybar_d - xbar_d'beta)`` and sampling fraction
    ``f_d = n_d / N_d`` (0 without ``popsize``) the EBLUP is ``f_d ybar_d +
    (Xbar_d - f_d xbar_d)'beta + (1 - f_d) u_d``; areas without sample get the
    synthetic ``Xbar_d'beta``. ``X`` includes the intercept column and
    ``xbar_pop[d]`` the matching population means (with the 1).

    References
    ----------
    Battese, G. E., Harter, R. M. and Fuller, W. A. (1988). An error-components
    model for prediction of county crop areas using survey and satellite
    data. *JASA*, 83(401), 28-36.

    Examples
    --------
    >>> y = [1.0, 2.0, 2.0, 3.0, 5.0, 6.0]
    >>> r = bhf_eblup(y, [[1]] * 6, [0, 0, 1, 1, 2, 2], [[1], [1], [1]])
    >>> round(r.sigma2_u, 10), round(r.sigma2_e, 10)
    (4.0833333333, 0.5)
    """
    if method not in ("REML", "ML"):
        raise ValueError("method must be REML or ML")
    yv = [float(v) for v in y]
    Xm = _mat(X)
    p = len(Xm[0])
    labels = list(areas) if areas is not None else sorted(set(area), key=list(area).index)
    groups = []
    for lab in labels:
        idx = [i for i, a in enumerate(area) if a == lab]
        if idx:
            groups.append(([Xm[i] for i in idx], [yv[i] for i in idx]))
    reml = method == "REML"
    lo, hi = -30.0, 15.0
    if _bhf_profile(math.exp(lo), groups, p, reml)[4] <= 0:
        rho = 0.0
    else:
        a, b = lo, hi
        for _ in range(200):
            mid = 0.5 * (a + b)
            if mid in (a, b):
                break
            if _bhf_profile(math.exp(mid), groups, p, reml)[4] > 0:
                a = mid
            else:
                b = mid
        rho = math.exp(0.5 * (a + b))
    ll, beta, s2e, Q, _ = _bhf_profile(rho, groups, p, reml)
    s2u = rho * s2e
    est, u, n_d = [], [], []
    for k, lab in enumerate(labels):
        idx = [i for i, a2 in enumerate(area) if a2 == lab]
        xb = [float(v) for v in xbar_pop[k]]
        if not idx:
            est.append(ssum(xb[t] * beta[t] for t in range(p)))
            u.append(0.0)
            n_d.append(0)
            continue
        n = len(idx)
        ybar = ssum(yv[i] for i in idx) / n
        xs = [ssum(Xm[i][t] for i in idx) / n for t in range(p)]
        gam = s2u / (s2u + s2e / n)
        ud = gam * (ybar - ssum(xs[t] * beta[t] for t in range(p)))
        f = n / float(popsize[k]) if popsize is not None else 0.0
        est.append(f * ybar + ssum((xb[t] - f * xs[t]) * beta[t] for t in range(p)) + (1.0 - f) * ud)
        u.append(ud)
        n_d.append(n)
    return RichResult(
        payload={
            "eblup": est,
            "random_effects": u,
            "beta": beta,
            "se_beta": [math.sqrt(s2e * Q[t][t]) for t in range(p)],
            "sigma2_u": s2u,
            "sigma2_e": s2e,
            "loglik": ll,
            "sample_sizes": n_d,
            "areas": labels,
            "method": method,
        }
    )


def marshall_eb(cases, population, *, family: str = "poisson") -> RichResult:
    r"""Marshall's (1991) global empirical-Bayes rates by the method of moments (as ``spdep::EBest``).

    ``b = sum n / sum x``, ``s^2 = sum x (r - b)^2 / sum x`` for raw rates
    ``r = n / x``. Poisson: prior variance ``a = max(0, s^2 - b / xbar)`` and
    ``r_EB = b + a (r - b) / (a + b / x)``. Binomial: the shrinkage factor
    ``rho`` of Marshall's binomial version.

    References
    ----------
    Marshall, R. J. (1991). Mapping disease and mortality rates using empirical
    Bayes estimators. *Applied Statistics*, 40(2), 283-294.

    Examples
    --------
    >>> r = marshall_eb([2, 8], [100, 100])
    >>> [round(v, 10) for v in r.estimate], round(r.a, 12)
    ([0.0366666667, 0.0633333333], 0.0004)
    """
    n = [float(v) for v in cases]
    x = [float(v) for v in population]
    if any(v <= 0 for v in x):
        raise ValueError("non-positive risk population")
    m = len(n)
    raw = [a / b for a, b in zip(n, x)]
    xs = ssum(x)
    b = ssum(n) / xs
    s2 = ssum(xi * (r - b) ** 2 / xs for xi, r in zip(x, raw))
    if family == "poisson":
        a = max(0.0, s2 - b / (xs / m))
        est = [b + a * (r - b) / (a + b / xi) for r, xi in zip(raw, x)]
    elif family == "binomial":
        xm = xs / m
        rho = [(xi * s2 - (xi / xm) * (b * (1 - b))) / ((xi - 1) * s2 + ((xm - xi) / xm) * (b * (1 - b))) for xi in x]
        est = [rh * r + (1 - rh) * b for rh, r in zip(rho, raw)]
        a = s2
    else:
        raise ValueError("family must be poisson or binomial")
    return RichResult(payload={"raw": raw, "estimate": est, "a": a, "b": b})


def potthoff_whittinghill(observed, expected) -> RichResult:
    r"""Potthoff-Whittinghill test of homogeneous relative risk against over-dispersion (as ``DCluster::pottwhitt.stat``).

    ``T = E_+ sum_i O_i (O_i - 1) / E_i`` with asymptotic mean ``O_+ (O_+ - 1)``
    and variance ``2 (k - 1) O_+ (O_+ - 1)`` under the multinomial null; the
    p-value is ``min(Phi(z), 1 - Phi(z))`` (DCluster's two-tail convention)
    and ``p_upper`` the one-sided over-dispersion p-value ``1 - Phi(z)``.

    References
    ----------
    Potthoff, R. F. and Whittinghill, M. (1966). Testing for homogeneity: II.
    The Poisson distribution. *Biometrika*, 53(1/2), 183-190.

    Examples
    --------
    >>> r = potthoff_whittinghill([3, 3, 3, 3], [3.0, 3.0, 3.0, 3.0])
    >>> r.T, r.mean, r.variance
    (96.0, 132.0, 792.0)
    """
    obs = [float(v) for v in observed]
    Ev = [float(v) for v in expected]
    T = ssum(Ev) * ssum(o * (o - 1) / e for o, e in zip(obs, Ev))
    tot = ssum(obs)
    mean = tot * (tot - 1)
    var = 2 * (len(obs) - 1) * mean
    z = (T - mean) / math.sqrt(var)
    lo = float(pnorm(z))
    return RichResult(
        payload={"T": T, "mean": mean, "variance": var, "z": z, "p_value": min(lo, 1 - lo), "p_upper": 1 - lo}
    )


def cheatsheet() -> str:
    return (
        "fay_herriot / bhf_eblup / marshall_eb / potthoff_whittinghill -> small-area EBLUPs, "
        "empirical-Bayes rates and rate-homogeneity test."
    )
