# morie.fn -- shared extreme-value core (rootcoder007/morie)
"""GEV and GPD primitives for the Coles shelf.

Everything here follows Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer:

* GEV distribution function, eq. (3.2):
  G(z) = exp{-[1 + xi (z-mu)/sigma]^{-1/xi}}, with the Gumbel limit
  G(z) = exp{-exp[-(z-mu)/sigma]} as xi -> 0.
* GEV log-likelihood, eq. (3.7)-(3.9).
* Return level z_p, eq. (3.4): z_p = mu - (sigma/xi)[1 - y_p^{-xi}],
  y_p = -log(1-p); Gumbel case eq. (3.5).
* GPD, eq. (4.2)-(4.4): H(y) = 1 - (1 + xi y / sigma_u)^{-1/xi}, its
  log-likelihood eq. (4.10), and the POT m-observation return level
  eq. (4.12)-(4.13): x_m = u + (sigma/xi)[(m zeta_u)^xi - 1].
* Delta-method variance for return levels, eq. (3.4 context) sec 3.3.3
  and sec. 4.3.3.

MLEs use the native Nelder-Mead from ``_sci_core.minimize`` on the
negative log-likelihood; covariance is the inverse of a central-
difference numeric Hessian (observed information), as Coles sec. 2.6.4
prescribes. The support constraint 1 + xi (z - mu)/sigma > 0 returns
-inf outside the feasible set (sec. 3.3.2 warning).

Page/equation numbers verified against the library PDF.

References
----------
Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme
Values*. Springer Series in Statistics. Springer.
ISBN 978-1-85233-459-8.

None of these PDFs is in the local library; they are cited from
bibliographic details, and the formulas here have not been
re-verified against them.
"""

from __future__ import annotations

import math

from . import _array_core as np
from . import _sci_core as sci

_XI_TINY = 1e-8      # |xi| below this uses the Gumbel/exponential limit


def _flat(x):
    a = np.asarray(x, dtype=float)
    return [float(v) for v in a.ravel()._flat()] \
        if hasattr(a, "ravel") else [float(v) for v in a._flat()]


# ------------------------------------------------------------- GEV
def gev_cdf(z, mu, sigma, xi):
    t = (z - mu) / sigma
    if abs(xi) < _XI_TINY:
        return math.exp(-math.exp(-t))
    arg = 1.0 + xi * t
    if arg <= 0.0:
        return 0.0 if xi > 0 else 1.0
    return math.exp(-arg ** (-1.0 / xi))


def gev_logpdf(z, mu, sigma, xi):
    if sigma <= 0:
        return float("-inf")
    t = (z - mu) / sigma
    if abs(xi) < _XI_TINY:
        return -math.log(sigma) - t - math.exp(-t)
    arg = 1.0 + xi * t
    if arg <= 0.0:
        return float("-inf")
    return (-math.log(sigma) - (1.0 + 1.0 / xi) * math.log(arg)
            - arg ** (-1.0 / xi))


def gev_loglik(x, mu, sigma, xi):
    return sum(gev_logpdf(v, mu, sigma, xi) for v in _flat(x))


def gev_quantile(p, mu, sigma, xi):
    """G^{-1}(p): eq. (3.4) with p the non-exceedance probability."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    yp = -math.log(p)
    if abs(xi) < _XI_TINY:
        return mu - sigma * math.log(yp)
    return mu + (sigma / xi) * (yp ** (-xi) - 1.0)


def gev_sample(n, mu, sigma, xi, rng):
    out = []
    for u in rng.uniform(0.0, 1.0, int(n))._flat():
        out.append(gev_quantile(float(u), mu, sigma, xi))
    return out


# ------------------------------------------------------------- GPD
def gpd_cdf(y, sigma, xi):
    if y < 0:
        return 0.0
    if abs(xi) < _XI_TINY:
        return 1.0 - math.exp(-y / sigma)
    arg = 1.0 + xi * y / sigma
    if arg <= 0.0:
        return 1.0
    return 1.0 - arg ** (-1.0 / xi)


def gpd_logpdf(y, sigma, xi):
    if sigma <= 0 or y < 0:
        return float("-inf")
    if abs(xi) < _XI_TINY:
        return -math.log(sigma) - y / sigma
    arg = 1.0 + xi * y / sigma
    if arg <= 0.0:
        return float("-inf")
    return -math.log(sigma) - (1.0 + 1.0 / xi) * math.log(arg)


def gpd_loglik(y, sigma, xi):
    return sum(gpd_logpdf(v, sigma, xi) for v in _flat(y))


def gpd_quantile(p, sigma, xi):
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    if abs(xi) < _XI_TINY:
        return -sigma * math.log(1.0 - p)
    return (sigma / xi) * ((1.0 - p) ** (-xi) - 1.0)


def gpd_sample(n, sigma, xi, rng):
    return [gpd_quantile(float(u), sigma, xi)
            for u in rng.uniform(0.0, 1.0, int(n))._flat()]


# ---------------------------------------------------------- fitting
def _hessian(f, theta, h=1e-4):
    k = len(theta)
    H = [[0.0] * k for _ in range(k)]
    f0 = f(theta)
    for i in range(k):
        for j in range(i, k):
            tpp = list(theta)
            tpm = list(theta)
            tmp = list(theta)
            tmm = list(theta)
            hi = h * max(1.0, abs(theta[i]))
            hj = h * max(1.0, abs(theta[j]))
            tpp[i] += hi
            tpp[j] += hj
            tpm[i] += hi
            tpm[j] -= hj
            tmp[i] -= hi
            tmp[j] += hj
            tmm[i] -= hi
            tmm[j] -= hj
            H[i][j] = H[j][i] = (f(tpp) - f(tpm) - f(tmp) + f(tmm)) \
                / (4.0 * hi * hj)
    del f0
    return H


def _inv(H):
    return np.linalg.pinv(np.marr(H)).tolist()


def gev_mle(x):
    """GEV MLE (mu, sigma, xi) with observed-information covariance.

    Start values per Coles sec. 3.3.2 practice: moment estimates under
    the Gumbel assumption (sigma0 = s*sqrt(6)/pi, mu0 = xbar -
    0.5772*sigma0), xi0 = 0.1.
    """
    xs = _flat(x)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two observations")
    xbar = sum(xs) / n
    s = math.sqrt(sum((v - xbar) ** 2 for v in xs) / (n - 1))
    sigma0 = s * math.sqrt(6.0) / math.pi
    mu0 = xbar - 0.5772156649015329 * sigma0

    def nll(th):
        mu, logs, xi = th
        return -gev_loglik(xs, mu, math.exp(logs), xi)

    res = sci.minimize(nll, [mu0, math.log(sigma0), 0.1],
                       method="Nelder-Mead",
                       options={"maxiter": 4000})
    mu, logs, xi = [float(v) for v in res.x]
    sigma = math.exp(logs)

    def nll_nat(th):
        return -gev_loglik(xs, th[0], th[1], th[2])

    H = _hessian(nll_nat, [mu, sigma, xi])
    cov = _inv(H)
    return {"mu": mu, "sigma": sigma, "xi": xi,
            "loglik": -float(res.fun), "cov": cov, "n": n,
            "converged": bool(getattr(res, "success", True))}


def gpd_mle(y):
    """GPD MLE (sigma, xi) over exceedances y >= 0 (Coles sec 4.3.2)."""
    ys = _flat(y)
    n = len(ys)
    if n < 2:
        raise ValueError("need at least two exceedances")
    ybar = sum(ys) / n
    s2 = sum((v - ybar) ** 2 for v in ys) / (n - 1)
    # method-of-moments start (Hosking & Wallis 1987):
    xi0 = 0.5 * (1.0 - ybar * ybar / s2)
    sigma0 = ybar * (1.0 - xi0) if xi0 < 1 else ybar
    sigma0 = max(sigma0, 1e-8)

    def nll(th):
        logs, xi = th
        return -gpd_loglik(ys, math.exp(logs), xi)

    res = sci.minimize(nll, [math.log(sigma0), xi0 if abs(xi0) < 0.9
                             else 0.1],
                       method="Nelder-Mead",
                       options={"maxiter": 4000})
    logs, xi = [float(v) for v in res.x]
    sigma = math.exp(logs)

    def nll_nat(th):
        return -gpd_loglik(ys, th[0], th[1])

    H = _hessian(nll_nat, [sigma, xi])
    cov = _inv(H)
    return {"sigma": sigma, "xi": xi, "loglik": -float(res.fun),
            "cov": cov, "n": n,
            "converged": bool(getattr(res, "success", True))}


# ----------------------------------------------------- return levels
def gev_return_level(T, mu, sigma, xi):
    """z_T with return period T: quantile at p = 1 - 1/T (eq. 3.4)."""
    if T <= 1:
        raise ValueError("return period T must exceed 1")
    return gev_quantile(1.0 - 1.0 / T, mu, sigma, xi)


def gev_return_level_grad(T, mu, sigma, xi):
    """Gradient of z_T in (mu, sigma, xi), Coles eq. below (3.4) /
    sec. 3.3.3 delta method."""
    yp = -math.log(1.0 - 1.0 / T)
    if abs(xi) < _XI_TINY:
        return [1.0, -math.log(yp), 0.0]
    d_mu = 1.0
    d_sigma = (yp ** (-xi) - 1.0) / xi
    d_xi = (-sigma / (xi * xi)) * (yp ** (-xi) - 1.0) \
        - (sigma / xi) * (yp ** (-xi)) * math.log(yp)
    return [d_mu, d_sigma, d_xi]


def pot_return_level(m, u, sigma, xi, zeta_u):
    """m-observation return level under POT, Coles eq. (4.12)-(4.13):
    x_m = u + (sigma/xi)[(m zeta_u)^xi - 1] (log form as xi -> 0)."""
    if m * zeta_u <= 1.0:
        raise ValueError("m * zeta_u must exceed 1 for a level above "
                         "the threshold")
    if abs(xi) < _XI_TINY:
        return u + sigma * math.log(m * zeta_u)
    return u + (sigma / xi) * ((m * zeta_u) ** xi - 1.0)
