# morie.fn -- function file (rootcoder007/morie)
"""Spatial epidemiology and disease surveillance: the Bayesian outbreak detector of the
``surveillance`` package, CUSUM surveillance of counts against expected values (per region, the
spatial CUSUM of Rogerson and Yamada), ecological regression of area counts with expected-count
offsets (Poisson, negative binomial and zero-inflated Poisson), the Leroux and BYM2 area-level
random-effect structures, and buffer and kernel exposure assessment."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import solve, ssum
from ._richresult import RichResult

__all__ = [
    "bayes_outbreak",
    "cusum_surveillance",
    "ecological_regression",
    "leroux_precision",
    "bym2_structure",
    "buffer_exposure",
    "kernel_exposure",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _mat(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in r] for r in a.tolist()]


def _qnbinom(p, size, prob):
    # smallest y with P(Y <= y) >= p for the negative binomial (size, prob), R's fuzz 1 - 64 eps
    target = p * (1 - 64 * 2.220446049250313e-16)
    f = math.exp(size * math.log(prob))
    cdf = f
    y = 0
    while cdf < target:
        f *= (y + size) / (y + 1) * (1 - prob)
        y += 1
        cdf += f
    return y


def bayes_outbreak(observed, freq=52, b=0, w=6, act_y=True, alpha=0.05, time_points=None):
    r"""Bayesian outbreak detection (``surveillance::algo.bayes``).

    For time ``t`` the reference values are the ``w`` preceding counts (if
    ``act_y``) and, for each of ``b`` previous years, the ``2w + 1`` counts
    around ``t - i freq``. With ``S`` their sum and ``n`` their number, a
    Poisson rate with the Jeffreys-type Gamma(1/2) prior gives the negative
    binomial predictive ``NB(S + 1/2, n / (n + 1))``; its ``1 - alpha``
    quantile is the upper bound and an alarm is raised when the count
    exceeds it. ``time_points`` are 1-based (default: every admissible one).

    References
    ----------
    Hoehle, M. (2007). surveillance: an R package for the monitoring of
    infectious diseases. *Computational Statistics* 22, 571-582.

    Riebler, A. (2004). *Empirischer Vergleich von statistischen Methoden zur
    Ausbruchserkennung bei Surveillance Daten*. Bachelor thesis, LMU Munich.

    Examples
    --------
    >>> r = bayes_outbreak([3, 5, 2, 4, 6, 3, 4, 12], w=6, time_points=[8])
    >>> r.upperbound, r.alarm
    ([8], [True])
    """
    x = _vec(observed)
    lo = b * freq + w + 1
    tps = list(range(lo, len(x) + 1)) if time_points is None else [int(t) for t in time_points]
    ub, al = [], []
    for t in tps:
        if t - b * freq - w < 1:
            raise ValueError("the series is too short for the reference window")
        base = x[t - w - 1 : t - 1] if act_y else []
        for i in range(1, b + 1):
            base = base + x[t - i * freq - w - 1 : t - i * freq + w]
        s = ssum(v for v in base if not math.isnan(v))
        n = sum(1 for v in base if not math.isnan(v))
        u = _qnbinom(1 - alpha, s + 0.5, n / (n + 1.0))
        ub.append(u)
        al.append(x[t - 1] > u)
    return RichResult(payload={"upperbound": ub, "alarm": al, "time_points": tps})


def cusum_surveillance(observed, expected=None, start=None, k=1.04, h=2.26, trans="standard", reset=False):
    r"""CUSUM surveillance of counts (``surveillance::algo.cusum``) and the spatial CUSUM.

    Counts are standardised against their expectation ``m`` -- by default the
    mean of the counts before ``start`` (1-based) -- as ``"standard"`` ``(x -
    m) / sqrt(m)``, ``"rossi"`` ``(x - 3m + 2 sqrt(x m)) / (2 sqrt(m))``,
    ``"anscombe"`` ``3/2 (x^(2/3) - m^(2/3)) / m^(1/6)`` or ``"none"``; then
    ``S_t = max(0, S_{t-1} + z_t - k)`` with an alarm when ``S_t >= h`` (and the
    sum reset to 0, and reported as 0, if ``reset``). With ``expected`` (one value per count, or a
    matrix times x regions matching ``observed``) each region is monitored
    against its own expected counts: the spatial surveillance CUSUM of
    Rogerson and Yamada (2004).

    References
    ----------
    Rossi, G., Lampugnani, L. and Marchi, M. (1999). An approximate CUSUM
    procedure for surveillance of health events. *Statistics in Medicine* 18,
    2111-2122.

    Rogerson, P. A. and Yamada, I. (2004). Approaches to syndromic
    surveillance when data consist of small regional counts. *MMWR* 53
    (Suppl.), 79-85.

    Examples
    --------
    >>> r = cusum_surveillance([4, 5, 3, 4, 9, 11, 4], start=5)
    >>> [round(v, 10) for v in r.cusum], r.alarm
    ([1.46, 3.92, 2.88], [False, True, True])
    """
    X = np.asarray(observed, dtype=float)
    multi = X.ndim == 2
    cols = _mat(observed) if multi else [[v] for v in _vec(observed)]
    ncol = len(cols[0])
    E = None
    if expected is not None:
        E = _mat(expected) if multi else [[v] for v in _vec(expected)]
    out_c, out_a = [], []
    for c in range(ncol):
        x = [r[c] for r in cols]
        if E is not None:
            idx = list(range(len(x)))
            m = [E[i][c] for i in idx]
        else:
            s0 = int(start)
            idx = list(range(s0 - 1, len(x)))
            base = ssum(x[: s0 - 1]) / (s0 - 1)
            m = [base] * len(idx)
        z = []
        for i, mm in zip(idx, m):
            v = x[i]
            if trans == "standard":
                z.append((v - mm) / math.sqrt(mm))
            elif trans == "rossi":
                z.append((v - 3 * mm + 2 * math.sqrt(v * mm)) / (2 * math.sqrt(mm)))
            elif trans == "anscombe":
                z.append(1.5 * (v ** (2 / 3) - mm ** (2 / 3)) / mm ** (1 / 6))
            elif trans == "none":
                z.append(v)
            else:
                raise ValueError("trans must be standard, rossi, anscombe or none")
        s, cs, al = 0.0, [], []
        for v in z:
            s = max(0.0, s + (v - k))
            a = s >= h
            if reset and a:
                s = 0.0
            cs.append(s)
            al.append(a)
        out_c.append(cs)
        out_a.append(al)
    if not multi:
        return RichResult(payload={"cusum": out_c[0], "alarm": out_a[0]})
    return RichResult(payload={"cusum": [list(r) for r in zip(*out_c)], "alarm": [list(r) for r in zip(*out_a)]})


def _irls_offset(y, X, off, wfun, maxit=100, tol=1e-12):
    k = len(X[0])
    beta = [0.0] * k
    beta[0] = math.log(max(ssum(y), 0.5) / ssum(math.exp(o) for o in off))
    for _ in range(maxit):
        eta = [ssum(r[a] * beta[a] for a in range(k)) + o for r, o in zip(X, off)]
        mu = [math.exp(e) for e in eta]
        w = wfun(mu)
        z = [e - o + (yy - m) / m for e, o, yy, m in zip(eta, off, y, mu)]
        A = [[ssum(w[i] * X[i][a] * X[i][b] for i in range(len(y))) for b in range(k)] for a in range(k)]
        new = solve(A, [ssum(w[i] * X[i][a] * z[i] for i in range(len(y))) for a in range(k)])
        if max(abs(p - q) for p, q in zip(new, beta)) < tol * (1 + max(abs(v) for v in beta)):
            beta = new
            break
        beta = new
    return beta


def _nb_ll(y, mu, th):
    return ssum(
        math.lgamma(yy + th)
        - math.lgamma(th)
        - math.lgamma(yy + 1)
        + th * math.log(th / (th + m))
        + yy * math.log(m / (th + m))
        for yy, m in zip(y, mu)
    )


def _digamma(x):
    acc = 0.0
    while x < 12.0:
        acc -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    i2 = inv * inv
    return acc + math.log(x) - 0.5 * inv - i2 * (1 / 12 - i2 * (1 / 120 - i2 * (1 / 252 - i2 / 240)))


def _trigamma(x):
    acc = 0.0
    while x < 12.0:
        acc += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    i2 = inv * inv
    return acc + inv + i2 / 2 + inv * i2 * (1 / 6 - i2 * (1 / 30 - i2 * (1 / 42 - i2 / 30)))


def ecological_regression(counts, expected, X=None, family="poisson", Z=None, tol=1e-12, maxit=500):
    r"""Ecological (area-level) regression of disease counts with an expected-count offset.

    ``log E(Y_i) = log e_i + x_i'beta``: ``family="poisson"`` by IRLS;
    ``"negbin"`` the NB2 model ``Var = mu + mu^2 / theta`` alternating IRLS for
    ``beta`` and Newton steps for ``theta`` (as ``MASS::glm.nb``); ``"zip"``
    the zero-inflated Poisson ``P(Y = 0) = pi + (1 - pi) exp(-mu)``,
    ``logit pi = z'gamma`` (``Z`` default intercept only), by EM (Lambert
    1992). ``X`` defaults to the intercept; the relative risks are
    ``exp(beta)``.

    References
    ----------
    Lambert, D. (1992). Zero-inflated Poisson regression, with an application
    to defects in manufacturing. *Technometrics* 34, 1-14.

    Venables, W. N. and Ripley, B. D. (2002). *Modern Applied Statistics with
    S*, 4th edn. Springer, section 7.4.

    Wakefield, J. (2008). Ecologic studies revisited. *Annual Review of Public
    Health* 29, 75-90.

    Examples
    --------
    >>> r = ecological_regression([4, 10, 3, 7], [5.0, 8.0, 4.0, 6.0], [[0.1], [0.9], [0.2], [0.5]])
    >>> [round(b, 10) for b in r.coefficients]
    [-0.2703623308, 0.5889757282]
    """
    y = _vec(counts)
    off = [math.log(v) for v in _vec(expected)]
    n = len(y)
    Xm = [[1.0] for _ in range(n)] if X is None else [[1.0] + r for r in _mat(X)]
    k = len(Xm[0])

    def pll(mu):
        return ssum(yy * math.log(m) - m - math.lgamma(yy + 1) for yy, m in zip(y, mu))

    if family == "poisson":
        beta = _irls_offset(y, Xm, off, lambda mu: mu)
        mu = [math.exp(ssum(r[a] * beta[a] for a in range(k)) + o) for r, o in zip(Xm, off)]
        return RichResult(payload={"coefficients": beta, "fitted": mu, "loglik": pll(mu)})
    if family == "negbin":
        beta = _irls_offset(y, Xm, off, lambda mu: mu)
        mu = [math.exp(ssum(r[a] * beta[a] for a in range(k)) + o) for r, o in zip(Xm, off)]
        th = n / ssum((yy / m - 1) ** 2 for yy, m in zip(y, mu))
        for _ in range(maxit):
            for _ in range(100):
                sc = ssum(
                    _digamma(th + yy) - _digamma(th) + math.log(th) + 1 - math.log(th + m) - (yy + th) / (m + th)
                    for yy, m in zip(y, mu)
                )
                inf = ssum(
                    -_trigamma(th + yy) + _trigamma(th) - 1 / th + 2 / (m + th) - (yy + th) / (m + th) ** 2
                    for yy, m in zip(y, mu)
                )
                if not inf > 0:
                    # no overdispersion: the likelihood increases towards the Poisson limit
                    th = min(2 * th, 1e10)
                    if th >= 1e10:
                        break
                    continue
                d = sc / inf
                th = max(th + d, 1e-8)
                if abs(d) < tol * (1 + th):
                    break
            t = th
            nb = _irls_offset(y, Xm, off, lambda mu, t=t: [m / (1 + m / t) for m in mu])
            done = max(abs(p - q) for p, q in zip(nb, beta)) < 1e-10
            beta = nb
            mu = [math.exp(ssum(r[a] * beta[a] for a in range(k)) + o) for r, o in zip(Xm, off)]
            if done:
                break
        return RichResult(payload={"coefficients": beta, "theta": th, "fitted": mu, "loglik": _nb_ll(y, mu, th)})
    if family == "zip":
        Zm = [[1.0] for _ in range(n)] if Z is None else [[1.0] + r for r in _mat(Z)]
        q = len(Zm[0])
        beta = _irls_offset(y, Xm, off, lambda mu: mu)
        gam = [0.0] * q
        ll_old = -math.inf
        for _ in range(maxit * 20):
            mu = [math.exp(ssum(r[a] * beta[a] for a in range(k)) + o) for r, o in zip(Xm, off)]
            pi = [1 / (1 + math.exp(-ssum(r[a] * gam[a] for a in range(q)))) for r in Zm]
            zz = [p / (p + (1 - p) * math.exp(-m)) if yy == 0 else 0.0 for yy, p, m in zip(y, pi, mu)]
            # M step: weighted Poisson for beta (weights 1 - z) and logistic regression of z for gamma
            wts = [1 - v for v in zz]
            b = beta
            for _ in range(50):
                eta = [ssum(r[a] * b[a] for a in range(k)) + o for r, o in zip(Xm, off)]
                m2 = [math.exp(e) for e in eta]
                A = [[ssum(wts[i] * m2[i] * Xm[i][a] * Xm[i][c] for i in range(n)) for c in range(k)] for a in range(k)]
                g = [ssum(wts[i] * (y[i] - m2[i]) * Xm[i][a] for i in range(n)) for a in range(k)]
                st = solve(A, g)
                b = [p + s for p, s in zip(b, st)]
                if max(abs(s) for s in st) < 1e-13:
                    break
            beta = b
            gm = gam
            for _ in range(50):
                p2 = [1 / (1 + math.exp(-ssum(r[a] * gm[a] for a in range(q)))) for r in Zm]
                A = [
                    [ssum(p2[i] * (1 - p2[i]) * Zm[i][a] * Zm[i][c] for i in range(n)) for c in range(q)]
                    for a in range(q)
                ]
                g = [ssum((zz[i] - p2[i]) * Zm[i][a] for i in range(n)) for a in range(q)]
                st = solve(A, g)
                gm = [p + s for p, s in zip(gm, st)]
                if max(abs(s) for s in st) < 1e-13:
                    break
            gam = gm
            mu = [math.exp(ssum(r[a] * beta[a] for a in range(k)) + o) for r, o in zip(Xm, off)]
            pi = [1 / (1 + math.exp(-ssum(r[a] * gam[a] for a in range(q)))) for r in Zm]
            ll = ssum(
                math.log(p + (1 - p) * math.exp(-m))
                if yy == 0
                else math.log(1 - p) + yy * math.log(m) - m - math.lgamma(yy + 1)
                for yy, p, m in zip(y, pi, mu)
            )
            if abs(ll - ll_old) < tol * (1 + abs(ll)):
                break
            ll_old = ll
        return RichResult(payload={"coefficients": beta, "zero_coefficients": gam, "fitted": mu, "loglik": ll})
    raise ValueError("family must be 'poisson', 'negbin' or 'zip'")


def leroux_precision(A, rho, tau=1.0):
    r"""Leroux, Lei and Breslow (2000) conditional autoregressive precision.

    ``Q = tau (rho (D - A) + (1 - rho) I)`` for a symmetric adjacency ``A``
    with row sums ``D``: an interpolation between independent (``rho = 0``) and
    intrinsic CAR (``rho = 1``) area effects; conditionally ``b_i | b_-i ~
    N(rho sum_j a_ij b_j / (rho d_i + 1 - rho), 1 / (tau (rho d_i + 1 -
    rho)))``.

    References
    ----------
    Leroux, B. G., Lei, X. and Breslow, N. (2000). Estimation of disease rates
    in small areas: a new mixed model for spatial dependence. In *Statistical
    Models in Epidemiology, the Environment, and Clinical Trials*, Springer,
    179-191.

    Examples
    --------
    >>> leroux_precision([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 0.5, 2.0)
    [[2.0, -1.0, 0.0], [-1.0, 3.0, -1.0], [0.0, -1.0, 2.0]]
    """
    Am = _mat(A)
    n = len(Am)
    d = [ssum(r) for r in Am]
    return [
        [tau * (rho * ((d[i] if i == j else 0.0) - Am[i][j]) + (1 - rho) * (1.0 if i == j else 0.0)) for j in range(n)]
        for i in range(n)
    ]


def bym2_structure(A, tau=1.0, phi=0.5, tol=1e-10):
    r"""BYM2 reparameterisation of the Besag-York-Mollie model (Riebler et al. 2016).

    With the intrinsic CAR precision ``Q = D - A`` and its generalized inverse
    ``Q^-`` (eigen-decomposition, dropping the null space of each connected
    component), the scaling factor ``s = exp(mean log diag(Q^-))`` makes the
    scaled ICAR ``u*`` have geometric-mean marginal variance 1; the BYM2 area
    effect ``b = (sqrt(1 - phi) v + sqrt(phi) u*) / sqrt(tau)`` has covariance
    ``((1 - phi) I + phi Q^- / s) / tau``: ``phi`` is the share of spatially
    structured variance.

    References
    ----------
    Riebler, A., Sorbye, S. H., Simpson, D. and Rue, H. (2016). An intuitive
    Bayesian spatial model for disease mapping that accounts for scaling.
    *Statistical Methods in Medical Research* 25, 1145-1165.

    Sorbye, S. H. and Rue, H. (2014). Scaling intrinsic Gaussian Markov random
    field priors in spatial modelling. *Spatial Statistics* 8, 39-51.

    Examples
    --------
    >>> r = bym2_structure([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> round(r.scaling_factor, 12)
    0.409336833182
    """
    Am = _mat(A)
    n = len(Am)
    Q = [[(ssum(Am[i]) if i == j else 0.0) - Am[i][j] for j in range(n)] for i in range(n)]
    w, v = np.linalg.eigh(np.asarray(Q, dtype=float))
    lam = [float(x) for x in w.tolist()]
    V = [[float(x) for x in r] for r in v.tolist()]
    mx = max(lam)
    keep = [m for m in range(n) if lam[m] > tol * mx]
    G = [[ssum(V[i][m] * V[j][m] / lam[m] for m in keep) for j in range(n)] for i in range(n)]
    s = math.exp(ssum(math.log(G[i][i]) for i in range(n)) / n)
    cov = [[((1 - phi) * (1.0 if i == j else 0.0) + phi * G[i][j] / s) / tau for j in range(n)] for i in range(n)]
    return RichResult(payload={"scaling_factor": s, "generalized_inverse": G, "covariance": cov})


def _dist(p, q, metric):
    if metric == "euclidean":
        return math.sqrt(ssum((a - b) ** 2 for a, b in zip(p, q)))
    la1, lo1, la2, lo2 = [math.radians(v) for v in (p[0], p[1], q[0], q[1])]
    a = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(a)))


def buffer_exposure(receptors, sources, radius, weights=None, metric="euclidean"):
    r"""Buffer-based exposure: sources (or their emission weights) within ``radius`` of each receptor.

    Returns the count and weighted sum of sources within the buffer and the
    nearest-source distance; ``metric="haversine"`` takes (lat, lon) in degrees
    and km.

    References
    ----------
    Nuckols, J. R., Ward, M. H. and Jarup, L. (2004). Using geographic
    information systems for exposure assessment in environmental
    epidemiology studies. *Environmental Health Perspectives* 112, 1007-1015.

    Examples
    --------
    >>> r = buffer_exposure([[0, 0], [5, 5]], [[1, 0], [0, 2], [6, 5]], 1.5, [2.0, 1.0, 4.0])
    >>> r.count, r.weighted, r.nearest
    ([1, 1], [2.0, 4.0], [1.0, 1.0])
    """
    R, S = _mat(receptors), _mat(sources)
    w = [1.0] * len(S) if weights is None else _vec(weights)
    cnt, ws, nn = [], [], []
    for p in R:
        d = [_dist(p, q, metric) for q in S]
        inside = [i for i, v in enumerate(d) if v <= radius]
        cnt.append(len(inside))
        ws.append(ssum(w[i] for i in inside))
        nn.append(min(d))
    return RichResult(payload={"count": cnt, "weighted": ws, "nearest": nn})


def kernel_exposure(receptors, sources, bandwidth, weights=None, kernel="gaussian"):
    r"""Kernel density exposure at receptors: ``sum_j w_j K(d_ij / h) / h^2``.

    ``K`` is the bivariate Gaussian ``exp(-u^2 / 2) / (2 pi)`` or the quartic
    ``3 / pi (1 - u^2)^2`` for ``u < 1``, so the exposure is a density of
    sources (per unit area) weighted by emissions.

    References
    ----------
    Silverman, B. W. (1986). *Density Estimation for Statistics and Data
    Analysis*. Chapman and Hall.

    Examples
    --------
    >>> [round(v, 12) for v in kernel_exposure([[0, 0]], [[1, 0], [0, 2]], 1.0)]
    [0.118071631932]
    """
    R, S = _mat(receptors), _mat(sources)
    w = [1.0] * len(S) if weights is None else _vec(weights)
    h = float(bandwidth)
    out = []
    for p in R:
        tot = 0.0
        for q, wj in zip(S, w):
            u = math.sqrt(ssum((a - b) ** 2 for a, b in zip(p, q))) / h
            if kernel == "gaussian":
                kv = math.exp(-u * u / 2) / (2 * math.pi)
            elif kernel == "quartic":
                kv = 3 / math.pi * (1 - u * u) ** 2 if u < 1 else 0.0
            else:
                raise ValueError("kernel must be 'gaussian' or 'quartic'")
            tot += wj * kv
        out.append(tot / (h * h))
    return out


def cheatsheet() -> str:
    return (
        "bayes_outbreak / cusum_surveillance / ecological_regression / leroux_precision / bym2_structure / "
        "buffer_exposure / kernel_exposure -> spatial epidemiology."
    )


# alias kept from the retired placeholder of the same name
bym2_model = bym2_structure

# alias kept from the retired placeholder of the same name
cusum_spatial = cusum_surveillance

# alias kept from the retired placeholder of the same name
ecological_nb = ecological_regression

# alias kept from the retired placeholder of the same name
ecological_reg = ecological_regression

# alias kept from the retired placeholder of the same name
ecological_zip = ecological_regression

# alias kept from the retired placeholder of the same name
leroux_model = leroux_precision
