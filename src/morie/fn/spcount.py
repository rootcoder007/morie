# morie.fn -- function file (rootcoder007/morie)
"""Spatial count regression: Poisson, negative binomial, zero-inflated Poisson and zero-inflated negative
binomial models with a spatially lagged (SAR) mean, the Lagrange multiplier test for the spatial lag,
information criteria and the BYM spatial variance fraction."""

from __future__ import annotations

import math

from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult
from ._s03core import digamma

__all__ = [
    "sar_poisson",
    "sar_negbin",
    "sar_zip",
    "sar_zinb",
    "sar_poisson_lm_test",
    "count_model_ic",
    "bym_variance_fraction",
]


def _mat(A):
    return [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in A]


def _lagged_design(X, W, rho):
    """(I - rho W)^{-1} X, column by column."""
    n, p = len(X), len(X[0])
    A = [[(1.0 if i == j else 0.0) - rho * W[i][j] for j in range(n)] for i in range(n)]
    cols = [solve(A, [X[i][k] for i in range(n)]) for k in range(p)]
    return [[cols[k][i] for k in range(p)] for i in range(n)]


def _wls(X, z, w):
    p = len(X[0])
    M = [[ssum(w[i] * X[i][a] * X[i][b] for i in range(len(X))) for b in range(p)] for a in range(p)]
    r = [ssum(w[i] * X[i][a] * z[i] for i in range(len(X))) for a in range(p)]
    return solve(M, r)


def _glm_count(y, X, prior=None, theta=None, beta=None):
    """IRLS for a log-link Poisson (theta None) or NB2 (known theta) GLM with prior weights."""
    n, p = len(y), len(X[0])
    pw = [1.0] * n if prior is None else prior
    b = [0.0] * p if beta is None else list(beta)
    if beta is None:
        m0 = ssum(pw[i] * y[i] for i in range(n)) / ssum(pw)
        b = _wls(X, [math.log(max(m0, 1e-10))] * n, pw)
    dev_old = math.inf
    for _ in range(200):
        eta = [ssum(X[i][a] * b[a] for a in range(p)) for i in range(n)]
        mu = [math.exp(v) for v in eta]
        w = [pw[i] * (mu[i] if theta is None else mu[i] / (1 + mu[i] / theta)) for i in range(n)]
        z = [eta[i] + (y[i] - mu[i]) / mu[i] for i in range(n)]
        b = _wls(X, z, w)
        mu = [math.exp(ssum(X[i][a] * b[a] for a in range(p))) for i in range(n)]
        dev = -2 * _loglik(y, mu, theta, pw)
        if abs(dev - dev_old) <= 1e-12 * (abs(dev) + 0.1):
            break
        dev_old = dev
    return b, mu


def _loglik(y, mu, theta, pw):
    s = 0.0
    for i in range(len(y)):
        if theta is None:
            li = (
                y[i] * math.log(mu[i]) - mu[i] - math.lgamma(y[i] + 1)
                if mu[i] > 0
                else (0.0 if y[i] == 0 else -math.inf)
            )
        else:
            li = (
                math.lgamma(y[i] + theta)
                - math.lgamma(theta)
                - math.lgamma(y[i] + 1)
                + theta * math.log(theta / (theta + mu[i]))
                + y[i] * math.log(mu[i] / (theta + mu[i]))
            )
        s += pw[i] * li
    return s


def _theta_ml(y, mu, pw):
    """ML dispersion of NB2 given the means, by bisection of the score on log theta."""

    def score(lt):
        t = math.exp(lt)
        return ssum(
            pw[i] * (digamma(y[i] + t) - digamma(t) + math.log(t) + 1 - math.log(t + mu[i]) - (y[i] + t) / (t + mu[i]))
            for i in range(len(y))
        )

    lo, hi = math.log(1e-4), math.log(1e6)
    slo = score(lo)
    if (slo > 0) == (score(hi) > 0):
        return math.exp(hi) if slo > 0 else math.exp(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        sm = score(mid)
        if (sm > 0) == (slo > 0):
            lo, slo = mid, sm
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return math.exp(0.5 * (lo + hi))


def _fit_nb(y, X, prior=None):
    n = len(y)
    pw = [1.0] * n if prior is None else prior
    b, mu = _glm_count(y, X, pw)
    theta = _theta_ml(y, mu, pw)
    ll_old = -math.inf
    for _ in range(100):
        b, mu = _glm_count(y, X, pw, theta, b)
        theta = _theta_ml(y, mu, pw)
        ll = _loglik(y, mu, theta, pw)
        if abs(ll - ll_old) <= 1e-12 * (abs(ll) + 0.1):
            break
        ll_old = ll
    return b, mu, theta


def _logit_fit(tau, Z, g=None):
    """Logistic regression with fractional responses tau (IRLS)."""
    n, q = len(tau), len(Z[0])
    g = [0.0] * q if g is None else list(g)
    for _ in range(200):
        eta = [ssum(Z[i][a] * g[a] for a in range(q)) for i in range(n)]
        pi = [1.0 / (1.0 + math.exp(-v)) for v in eta]
        w = [max(p * (1 - p), 1e-12) for p in pi]
        z = [eta[i] + (tau[i] - pi[i]) / w[i] for i in range(n)]
        new = _wls(Z, z, w)
        done = max(abs(a - b) for a, b in zip(new, g)) < 1e-12
        g = new
        if done:
            break
    return g


def _zi_fit(y, X, Z, nb):
    """EM for zero-inflated Poisson / NB2 with a logit zero model."""
    n = len(y)
    q = len(Z[0])
    b, mu = _glm_count(y, X)
    theta = None
    g = [0.0] * q
    ll_old = -math.inf
    for _ in range(1000):
        pi = [1.0 / (1.0 + math.exp(-ssum(Z[i][a] * g[a] for a in range(q)))) for i in range(n)]
        p0 = [math.exp(-mu[i]) if theta is None else (theta / (theta + mu[i])) ** theta for i in range(n)]
        tau = [pi[i] / (pi[i] + (1 - pi[i]) * p0[i]) if y[i] == 0 else 0.0 for i in range(n)]
        g = _logit_fit(tau, Z, g)
        wt = [1.0 - t for t in tau]
        if nb:
            b, mu = _glm_count(y, X, wt, theta if theta is not None else 1.0, b)
            theta = _theta_ml(y, mu, wt)
            b, mu = _glm_count(y, X, wt, theta, b)
        else:
            b, mu = _glm_count(y, X, wt, None, b)
        pi = [1.0 / (1.0 + math.exp(-ssum(Z[i][a] * g[a] for a in range(q)))) for i in range(n)]
        ll = _zi_loglik(y, mu, pi, theta)
        if abs(ll - ll_old) <= 1e-11 * (abs(ll) + 0.1):
            break
        ll_old = ll
    return b, g, theta, ll


def _zi_loglik(y, mu, pi, theta):
    s = 0.0
    for i in range(len(y)):
        if theta is None:
            p0 = math.exp(-mu[i])
            lc = y[i] * math.log(mu[i]) - mu[i] - math.lgamma(y[i] + 1)
        else:
            p0 = (theta / (theta + mu[i])) ** theta
            lc = (
                math.lgamma(y[i] + theta)
                - math.lgamma(theta)
                - math.lgamma(y[i] + 1)
                + theta * math.log(theta / (theta + mu[i]))
                + y[i] * math.log(mu[i] / (theta + mu[i]))
            )
        s += math.log(pi[i] + (1 - pi[i]) * p0) if y[i] == 0 else math.log(1 - pi[i]) + lc
    return s


def _golden(f, lo, hi):
    gr = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(200):
        if f1 >= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = f(x2)
        if hi - lo < 1e-9:
            break
    return 0.5 * (lo + hi)


def _prep(y, X, W):
    yv = [float(v) for v in y]
    Xm = _mat(X)
    Wm = _mat(W)
    if any(v < 0 or v != int(v) for v in yv):
        raise ValueError("y must be non-negative counts")
    if len(Xm) != len(yv) or len(Wm) != len(yv):
        raise ValueError("X and W must have n rows")
    return yv, Xm, Wm


def sar_poisson(y, X, W, *, rho_bounds=(-0.99, 0.99)) -> RichResult:
    r"""Spatial-lag Poisson regression ``E y = exp((I - rho W)^{-1} X beta)`` by profile maximum likelihood.

    For fixed ``rho`` the model is a Poisson GLM on the spatially filtered
    design ``(I - rho W)^{-1} X`` (fitted by IRLS); ``rho`` maximises the
    profile log-likelihood over ``rho_bounds`` (golden section). ``X`` should
    contain an intercept column; ``W`` is typically row-standardised.

    References
    ----------
    Lambert, D. M., Brown, J. P. and Florax, R. J. G. M. (2010). A two-step
    estimator for a spatial lag model of counts: theory, small sample
    performance and an application. *Regional Science and Urban Economics*, 40, 241-252.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> r = sar_poisson([1, 3, 4, 7], [[1, 0.1], [1, 0.4], [1, 0.5], [1, 0.9]], W)
    >>> round(r.loglik, 6) >= round(sar_poisson([1, 3, 4, 7], [[1, 0.1], [1, 0.4], [1, 0.5], [1, 0.9]], W, rho_bounds=(0, 0)).loglik, 6)
    True
    """
    yv, Xm, Wm = _prep(y, X, W)
    n = len(yv)

    def prof(rho):
        Xt = _lagged_design(Xm, Wm, rho)
        b, mu = _glm_count(yv, Xt)
        return _loglik(yv, mu, None, [1.0] * n)

    rho = rho_bounds[0] if rho_bounds[0] == rho_bounds[1] else _golden(prof, *rho_bounds)
    Xt = _lagged_design(Xm, Wm, rho)
    b, mu = _glm_count(yv, Xt)
    ll = _loglik(yv, mu, None, [1.0] * n)
    return RichResult(payload={"coefficients": b, "rho": rho, "fitted": mu, "loglik": ll, "k": len(b) + 1})


def sar_negbin(y, X, W, *, rho_bounds=(-0.99, 0.99)) -> RichResult:
    r"""Spatial-lag negative binomial (NB2) regression by profile maximum likelihood over ``rho``.

    Mean ``exp((I - rho W)^{-1} X beta)`` and variance ``mu + mu^2 / theta``;
    for fixed ``rho`` beta and theta alternate (IRLS for beta, ML for theta by
    bisection of the score), as ``MASS::glm.nb`` on the filtered design.

    References
    ----------
    Lambert, Brown and Florax (2010); Venables, W. N. and Ripley, B. D.
    (2002). *Modern Applied Statistics with S*, 4th edn. Springer, section 7.4.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
    >>> r = sar_negbin([0, 4, 1, 9, 3], [[1, 0.1], [1, 0.4], [1, 0.5], [1, 0.9], [1, 0.2]], W)
    >>> r.theta > 0
    True
    """
    yv, Xm, Wm = _prep(y, X, W)

    def prof(rho):
        b, mu, th = _fit_nb(yv, _lagged_design(Xm, Wm, rho))
        return _loglik(yv, mu, th, [1.0] * len(yv))

    rho = rho_bounds[0] if rho_bounds[0] == rho_bounds[1] else _golden(prof, *rho_bounds)
    b, mu, th = _fit_nb(yv, _lagged_design(Xm, Wm, rho))
    ll = _loglik(yv, mu, th, [1.0] * len(yv))
    return RichResult(payload={"coefficients": b, "rho": rho, "theta": th, "fitted": mu, "loglik": ll, "k": len(b) + 2})


def _zi(y, X, W, Z, rho_bounds, nb):
    yv, Xm, Wm = _prep(y, X, W)
    Zm = [[1.0] for _ in yv] if Z is None else _mat(Z)

    def prof(rho):
        return _zi_fit(yv, _lagged_design(Xm, Wm, rho), Zm, nb)[3]

    rho = rho_bounds[0] if rho_bounds[0] == rho_bounds[1] else _golden(prof, *rho_bounds)
    b, g, th, ll = _zi_fit(yv, _lagged_design(Xm, Wm, rho), Zm, nb)
    out = {"count_coefficients": b, "zero_coefficients": g, "rho": rho, "loglik": ll, "k": len(b) + len(g) + 1}
    if nb:
        out["theta"] = th
        out["k"] += 1
    return RichResult(payload=out)


def sar_zip(y, X, W, *, Z=None, rho_bounds=(-0.99, 0.99)) -> RichResult:
    r"""Spatial-lag zero-inflated Poisson regression (EM for fixed ``rho``, profile likelihood over ``rho``).

    ``P(y = 0) = pi + (1 - pi) e^{-mu}``, ``P(y = k) = (1 - pi) Pois(k; mu)``,
    ``logit pi = Z gamma`` (default intercept only), ``log mu`` the spatially
    lagged linear predictor ``(I - rho W)^{-1} X beta``. The EM algorithm
    (Lambert 1992) alternates weighted logistic and Poisson IRLS fits.

    References
    ----------
    Lambert, D. (1992). Zero-inflated Poisson regression, with an application
    to defects in manufacturing. *Technometrics*, 34, 1-14.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [0.5, 0, 0.5, 0, 0, 0], [0, 0.5, 0, 0.5, 0, 0], [0, 0, 0.5, 0, 0.5, 0], [0, 0, 0, 0.5, 0, 0.5], [0, 0, 0, 0, 1, 0]]
    >>> r = sar_zip([0, 0, 3, 5, 0, 2], [[1, 0.1], [1, 0.3], [1, 0.5], [1, 0.9], [1, 0.2], [1, 0.4]], W, rho_bounds=(0, 0))
    >>> 0 < 1 / (1 + math.exp(-r.zero_coefficients[0])) < 1
    True
    """
    return _zi(y, X, W, Z, rho_bounds, False)


def sar_zinb(y, X, W, *, Z=None, rho_bounds=(-0.99, 0.99)) -> RichResult:
    r"""Spatial-lag zero-inflated negative binomial regression (EM with an NB2 count part).

    As :func:`sar_zip` with the count part negative binomial (dispersion
    ``theta`` updated by weighted ML in each M-step).

    References
    ----------
    Greene, W. H. (1994). Accounting for excess zeros and sample selection in
    Poisson and negative binomial regression models. NYU Working Paper EC-94-10.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [0.5, 0, 0.5, 0, 0, 0], [0, 0.5, 0, 0.5, 0, 0], [0, 0, 0.5, 0, 0.5, 0], [0, 0, 0, 0.5, 0, 0.5], [0, 0, 0, 0, 1, 0]]
    >>> r = sar_zinb([0, 0, 3, 9, 0, 2], [[1, 0.1], [1, 0.3], [1, 0.5], [1, 0.9], [1, 0.2], [1, 0.4]], W, rho_bounds=(0, 0))
    >>> r.theta > 0
    True
    """
    return _zi(y, X, W, Z, rho_bounds, True)


def sar_poisson_lm_test(y, X, W) -> RichResult:
    r"""Lagrange multiplier (score) test of ``rho = 0`` in the spatial-lag Poisson model.

    Under ``rho = 0`` (an ordinary Poisson GLM with fitted ``mu``,
    ``eta = X beta``) the score is ``s = (y - mu)' W eta`` and the efficient
    information ``I = d' D d - d' D X (X' D X)^{-1} X' D d`` with ``d = W eta``
    and ``D = diag(mu)``; ``LM = s^2 / I`` is chi-square(1).

    References
    ----------
    Lambert, Brown and Florax (2010), section 2.3.
    Rao, C. R. (1948). Large sample tests of statistical hypotheses. *Proc.
    Cambridge Philosophical Society*, 44, 50-57.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> r = sar_poisson_lm_test([1, 3, 4, 7], [[1, 0.1], [1, 0.4], [1, 0.5], [1, 0.9]], W)
    >>> 0 <= r.p_value <= 1
    True
    """
    yv, Xm, Wm = _prep(y, X, W)
    n, p = len(yv), len(Xm[0])
    b, mu = _glm_count(yv, Xm)
    eta = [ssum(Xm[i][a] * b[a] for a in range(p)) for i in range(n)]
    d = [ssum(Wm[i][j] * eta[j] for j in range(n)) for i in range(n)]
    s = ssum((yv[i] - mu[i]) * d[i] for i in range(n))
    XDX = [[ssum(mu[i] * Xm[i][a] * Xm[i][c] for i in range(n)) for c in range(p)] for a in range(p)]
    XDd = [ssum(mu[i] * Xm[i][a] * d[i] for i in range(n)) for a in range(p)]
    Ai = inverse(XDX)
    info = ssum(mu[i] * d[i] * d[i] for i in range(n)) - ssum(
        XDd[a] * Ai[a][c] * XDd[c] for a in range(p) for c in range(p)
    )
    lm = s * s / info
    return RichResult(payload={"statistic": lm, "df": 1, "p_value": math.erfc(math.sqrt(lm / 2.0)), "score": s})


def count_model_ic(loglik: float, k: int, n: int) -> RichResult:
    r"""Information criteria of a fitted count model: ``AIC = -2 l + 2k``, ``BIC = -2 l + k log n``, ``AICc``.

    ``k`` counts every estimated parameter (coefficients, ``rho``, ``theta``, zero-model terms).

    References
    ----------
    Akaike, H. (1974). A new look at the statistical model identification.
    *IEEE Transactions on Automatic Control*, 19, 716-723.
    Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6, 461-464.

    Examples
    --------
    >>> r = count_model_ic(-10.0, 3, 20)
    >>> r.AIC, round(r.BIC, 6), r.AICc
    (26.0, 28.987197, 27.5)
    """
    aic = -2 * loglik + 2 * k
    return RichResult(
        payload={"AIC": aic, "BIC": -2 * loglik + k * math.log(n), "AICc": aic + 2 * k * (k + 1) / (n - k - 1)}
    )


def bym_variance_fraction(var_spatial: float, var_unstructured: float, *, Q=None) -> RichResult:
    r"""Fraction of the BYM random-effect variance that is spatial (CAR versus unstructured).

    ``phi = s v_s / (s v_s + v_u)``; with the ICAR structure matrix ``Q`` the
    spatial variance is scaled by ``s``, the geometric mean of the marginal
    variances ``diag(Q^+)`` (generalised inverse of the connected graph's
    ``Q``), so that the fraction refers to comparable typical variances
    (Sorbye and Rue 2014; Riebler et al. 2016). Without ``Q``, ``s = 1``.

    References
    ----------
    Sorbye, S. H. and Rue, H. (2014). Scaling intrinsic Gaussian Markov
    random field priors in spatial modelling. *Spatial Statistics*, 8, 39-51.
    Riebler, A., Sorbye, S. H., Simpson, D. and Rue, H. (2016). An intuitive
    Bayesian spatial model for disease mapping that accounts for scaling.
    *Statistical Methods in Medical Research*, 25, 1145-1165.

    Examples
    --------
    >>> bym_variance_fraction(1.0, 3.0).fraction
    0.25
    """
    s = 1.0
    if Q is not None:
        Qm = _mat(Q)
        n = len(Qm)
        # generalised inverse of a connected ICAR precision: (Q + 11'/n)^{-1} - 11'/n
        Qi = inverse([[Qm[i][j] + 1.0 / n for j in range(n)] for i in range(n)])
        diag = [Qi[i][i] - 1.0 / n for i in range(n)]
        s = math.exp(ssum(math.log(v) for v in diag) / n)
    return RichResult(payload={"fraction": s * var_spatial / (s * var_spatial + var_unstructured), "scale": s})


def cheatsheet() -> str:
    return (
        "sar_poisson / sar_negbin / sar_zip / sar_zinb / sar_poisson_lm_test / count_model_ic / "
        "bym_variance_fraction -> spatial count regression."
    )
