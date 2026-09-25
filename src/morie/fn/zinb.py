# morie.fn -- function file (rootcoder007/morie)
"""Zero-Inflated Negative Binomial (ZINB) model."""

from __future__ import annotations

import math

from . import _array_core as np
from ._containers import DescriptiveResult
from ._stats_core import _digamma


def _loglik_grad(theta, y, X):
    """ZINB2 log-likelihood and its gradient in (gamma, beta, log alpha).

    pi = expit(gamma), mu_i = exp(x_i beta), r = 1/alpha. A zero comes
    from the structural-zero class with probability pi or from the NB2
    count with probability (1 - pi) (r / (r + mu))^r.
    """
    n = len(y)
    k = len(X[0])
    g = theta[0]
    beta = theta[1:1 + k]
    la = theta[1 + k]
    r = math.exp(-la)
    # log pi and log(1 - pi) without overflow for large |gamma|
    lpi = -math.log1p(math.exp(-g)) if g > -30 else g - math.log1p(math.exp(g))
    l1pi = -math.log1p(math.exp(g)) if g < 30 else -g - math.log1p(math.exp(-g))
    pi = math.exp(lpi)
    ll = 0.0
    dg = 0.0
    db = [0.0] * k
    dr = 0.0
    for i in range(n):
        eta = sum(X[i][j] * beta[j] for j in range(k))
        mu = math.exp(eta)
        # log(r / (r + mu)) as -log1p(mu / r): exact as alpha -> 0, where
        # r is huge and the plain ratio rounds to 1
        lp = -math.log1p(mu / r)
        yi = y[i]
        if yi == 0:
            l0 = r * lp
            # log(pi + (1 - pi) e^l0) by log-sum-exp
            a, b = lpi, l1pi + l0
            m = a if a > b else b
            li = m + math.log(math.exp(a - m) + math.exp(b - m))
            w0 = math.exp(b - li)          # posterior share of the count class
            ll += li
            dg += (1.0 - w0) - pi          # d/dgamma
            deta = w0 * (-mu * r / (r + mu))
            dr += w0 * (lp + mu / (r + mu))
        else:
            # lgamma(y + r) - lgamma(r) and digamma(y + r) - digamma(r)
            # as finite sums: the lgamma difference of two numbers near
            # 1e13 loses every digit once alpha is small
            iy = int(yi)
            if iy <= 10000:
                lg = math.fsum(math.log(r + j) for j in range(iy))
                dgm = math.fsum(1.0 / (r + j) for j in range(iy))
            else:
                lg = math.lgamma(yi + r) - math.lgamma(r)
                dgm = _digamma(yi + r) - _digamma(r)
            ll += (l1pi + lg - math.lgamma(yi + 1.0) + r * lp
                   + yi * (eta - math.log(r + mu)))
            dg += -pi
            deta = (yi - mu) * r / (r + mu)
            dr += dgm + lp + (mu - yi) / (r + mu)
        for j in range(k):
            db[j] += deta * X[i][j]
    # r = exp(-log alpha), so d/dlog(alpha) = -r d/dr
    return ll, [dg] + db + [-r * dr]


def zero_inflated_negbin(
    y_counts: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> DescriptiveResult:
    """ZINB2 regression by maximum likelihood.

    Count part: NB2 with log link, mu_i = exp(b0 + x_i b), variance
    mu + alpha mu^2. Inflation part: a constant zero probability
    pi = expit(gamma). This is statsmodels'
    ``ZeroInflatedNegativeBinomialP(y, [1, X], exog_infl=ones, p=2)``
    and pscl's ``zeroinfl(y ~ X | 1, dist = "negbin")``.

    The previous version never used `X`, fitted a single constant mean,
    and took alpha from a method-of-moments step, so its log-likelihood
    was not a maximum of anything. Newton's method is now run on the
    exact score, with a central-difference Hessian of that score and
    step halving on the likelihood, until the score vanishes.

    Parameters
    ----------
    y_counts : (n,) counts
    X : (n, p) covariates; the intercept is added.
    max_iter : int
    tol : float
        Convergence threshold on the largest absolute score.

    Returns
    -------
    DescriptiveResult
        ``value`` is -2 logL; ``extra`` holds ``coefficients``,
        ``inflation_logit``, ``zero_prob``, ``alpha``, ``mu`` (mean fitted
        count mean), ``log_likelihood``, ``score_max``, ``converged``,
        ``n``, ``n_zeros``.
    """
    y = [float(v) for v in np.asarray(y_counts, dtype=float).ravel().tolist()]
    if any(v < 0 or v != math.floor(v) for v in y):
        raise ValueError("y_counts must be non-negative integers")
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    rows = Xa.tolist()
    n = len(y)
    if len(rows) != n:
        raise ValueError("X must have one row per count")
    Xi = [[1.0] + [float(v) for v in r] for r in rows]
    k = len(Xi[0])
    nz = sum(1 for v in y if v == 0)
    if nz == 0:
        raise ValueError("no zeros: the inflation probability is not "
                         "identified")
    ybar = sum(y) / n
    theta = [math.log(max(nz / n * 0.5, 1e-3) / (1 - max(nz / n * 0.5, 1e-3))),
             math.log(max(ybar, 1e-3))] + [0.0] * (k - 1) + [0.0]
    ll, g = _loglik_grad(theta, y, Xi)
    converged = False
    m = len(theta)
    for _ in range(int(max_iter)):
        if max(abs(v) for v in g) < tol:
            converged = True
            break
        # Hessian by central differences of the analytic score
        H = [[0.0] * m for _ in range(m)]
        for a in range(m):
            h = 1e-5 * max(1.0, abs(theta[a]))
            tp = theta[:]
            tm = theta[:]
            tp[a] += h
            tm[a] -= h
            gp = _loglik_grad(tp, y, Xi)[1]
            gm = _loglik_grad(tm, y, Xi)[1]
            for b in range(m):
                H[a][b] = (gp[b] - gm[b]) / (2.0 * h)
        Hs = [[0.5 * (H[a][b] + H[b][a]) for b in range(m)] for a in range(m)]
        try:
            step = np.linalg.solve(np.array(Hs), np.array(g)).tolist()
            step = [-v for v in step]
        except Exception:
            step = g[:]
        # a Newton step must go uphill; fall back to the gradient if not
        if sum(s * v for s, v in zip(step, g)) <= 0:
            step = g[:]
        t = 1.0
        while t > 1e-12:
            cand = [a + t * s for a, s in zip(theta, step)]
            try:
                lc, gc = _loglik_grad(cand, y, Xi)
            except (OverflowError, ValueError):
                lc = -math.inf
            if lc >= ll - 1e-12 * abs(ll):
                break
            t *= 0.5
        if t <= 1e-12:
            break
        theta, ll, g = cand, lc, gc
    if not converged and max(abs(v) for v in g) < tol:
        converged = True
    gam = theta[0]
    beta = theta[1:1 + k]
    alpha = math.exp(theta[1 + k])
    pi = 1.0 / (1.0 + math.exp(-gam))
    mus = [math.exp(sum(Xi[i][j] * beta[j] for j in range(k))) for i in range(n)]
    coef_names = ["intercept"] + [f"x{j}" for j in range(k - 1)]
    return DescriptiveResult(
        name="zinb",
        value=float(-2 * ll),
        extra={
            "coefficients": dict(zip(coef_names, beta)),
            "inflation_logit": float(gam),
            "zero_prob": float(pi),
            "alpha": float(alpha),
            "mu": float(sum(mus) / n),
            "log_likelihood": float(ll),
            "score_max": float(max(abs(v) for v in g)),
            "converged": converged,
            "n": n,
            "n_zeros": nz,
        },
    )


zinb = zero_inflated_negbin


def cheatsheet() -> str:
    return "zero_inflated_negbin({}) -> Zero-Inflated Negative Binomial (ZINB) model."
