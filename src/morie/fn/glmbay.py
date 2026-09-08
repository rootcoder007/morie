# morie.fn -- function file (rootcoder007/morie)
r"""Bayesian GLM by the Laplace approximation to the posterior.

A maximum-likelihood GLM reports a standard error from the curvature at
the mode and calls the interval a confidence interval. The Bayesian
version with a normal prior does very nearly the same arithmetic and
calls it a posterior -- the difference that matters is not the label but
the PRIOR, which regularises the very cases where ML breaks: under
separation the ML coefficient diverges to infinity while the posterior
mode stays finite, and that is visible here rather than being an error
message.

Fitting is penalised IRLS to the posterior mode, and the posterior is
then approximated by the Gaussian matching that mode and its curvature,

.. math:: \log p(y) \approx \ell(\hat\beta) + \log p(\hat\beta)
          + \tfrac{p}{2}\log 2\pi - \tfrac12\log|H|,

which gives the marginal likelihood used for comparison. The
approximation is exact for the Gaussian family and good in proportion to
how nearly quadratic the log-posterior is, so ``family`` is reported with
the result rather than treated as incidental.

References
----------
Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. and
Rubin, D. B. (2013) *Bayesian Data Analysis*, 3rd ed., CRC Press, Ch. 16
(hierarchical generalized linear models) and Sec. 4.1 (the normal
approximation to the posterior and its justification).

Gelman, A., Jakulin, A., Pittau, M. G. and Su, Y.-S. (2008) "A weakly
informative default prior distribution for logistic and other regression
models", *Annals of Applied Statistics* **2**(4), 1360-1383,
doi:10.1214/08-AOAS191. Why a proper prior is what keeps separation from
producing infinite estimates.

McCullagh, P. and Nelder, J. A. (1989) *Generalized Linear Models*,
2nd ed., Chapman & Hall, Ch. 2 (IRLS).
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["bayesian_glm"]

_EPS = 1e-12


def _links(family):
    if family == "binomial":
        def inv(e):
            e = max(-500.0, min(500.0, e))
            return 1.0 / (1.0 + math.exp(-e))

        def var(m):
            return max(m * (1.0 - m), 1e-10)

        def ll(y, m):
            m = min(max(m, 1e-12), 1.0 - 1e-12)
            return y * math.log(m) + (1.0 - y) * math.log(1.0 - m)
        return inv, var, ll
    if family == "poisson":
        def inv(e):
            return math.exp(max(-500.0, min(500.0, e)))

        def var(m):
            return max(m, 1e-10)

        def ll(y, m):
            m = max(m, 1e-12)
            return y * math.log(m) - m - math.lgamma(y + 1.0)
        return inv, var, ll
    if family == "gaussian":
        def inv(e):
            return e

        def var(m):
            return 1.0

        def ll(y, m):
            return -0.5 * (math.log(2.0 * math.pi) + (y - m) ** 2)
        return inv, var, ll
    raise ValueError("glmbay: family must be binomial, poisson or gaussian, "
                     "got %r" % (family,))


def bayesian_glm(X, y, family="binomial", prior_sd=2.5, add_intercept=True,
                 max_iter=100, tol=1e-10):
    r"""Posterior mode and Laplace covariance for a GLM with a normal prior."""
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    yv = [float(v) for v in k.vec(y)]
    n = len(Xm)
    if n == 0:
        raise ValueError("glmbay: no observations")
    if len(yv) != n:
        raise ValueError("glmbay: %d rows but %d responses" % (n, len(yv)))
    if add_intercept:
        Xm = [[1.0] + r for r in Xm]
    p = len(Xm[0])
    inv, var, ll = _links(family)
    ps = float(prior_sd)
    if ps <= 0.0:
        raise ValueError("glmbay: the prior standard deviation must be "
                         "positive")
    tau = 1.0 / (ps * ps)

    beta = [0.0] * p
    it, converged = 0, False
    H = None
    for it in range(1, int(max_iter) + 1):
        eta = [sum(Xm[i][a] * beta[a] for a in range(p)) for i in range(n)]
        mu = [inv(e) for e in eta]
        w = [var(m) for m in mu]
        z = [eta[i] + (yv[i] - mu[i]) / w[i] for i in range(n)]
        A = [[sum(w[i] * Xm[i][a] * Xm[i][b] for i in range(n))
              for b in range(p)] for a in range(p)]
        for a in range(p):
            A[a][a] += tau                      # the prior's contribution
        b = [sum(w[i] * Xm[i][a] * z[i] for i in range(n)) for a in range(p)]
        new = k.cholsolve(A, b)
        shift = max(abs(new[a] - beta[a]) for a in range(p))
        beta = new
        H = A
        if shift < tol:
            converged = True
            break

    cov = []
    for a in range(p):
        e = [1.0 if b2 == a else 0.0 for b2 in range(p)]
        cov.append(k.cholsolve(H, e))
    se = [math.sqrt(max(cov[a][a], 0.0)) for a in range(p)]

    eta = [sum(Xm[i][a] * beta[a] for a in range(p)) for i in range(n)]
    mu = [inv(e) for e in eta]
    loglik = sum(ll(yv[i], mu[i]) for i in range(n))
    logprior = sum(-0.5 * tau * beta[a] ** 2
                   - 0.5 * math.log(2.0 * math.pi * ps * ps)
                   for a in range(p))
    # log|H| from the Cholesky factor of H, without forming a determinant
    L = k.cholsolve  # marker: H is used only through solves above
    logdet = 0.0
    Hc = [row[:] for row in H]
    for a in range(p):                    # in-place Cholesky for log|H|
        s = Hc[a][a] - sum(Hc[a][u] ** 2 for u in range(a))
        s = max(s, 1e-300)
        Hc[a][a] = math.sqrt(s)
        for b2 in range(a + 1, p):
            Hc[b2][a] = ((Hc[b2][a]
                          - sum(Hc[b2][u] * Hc[a][u] for u in range(a)))
                         / Hc[a][a])
        logdet += 2.0 * math.log(Hc[a][a])
    log_marginal = (loglik + logprior + 0.5 * p * math.log(2.0 * math.pi)
                    - 0.5 * logdet)

    return RichResult(payload={
        "estimate": beta, "coefficients": beta, "posterior_sd": se,
        "std_error": se,
        "ci_lower": [beta[a] - 1.959963984540054 * se[a] for a in range(p)],
        "ci_upper": [beta[a] + 1.959963984540054 * se[a] for a in range(p)],
        "fitted": mu, "linear_predictor": eta,
        "loglik": loglik, "log_prior": logprior,
        "log_marginal": log_marginal, "log_det_hessian": logdet,
        "iterations": it, "converged": converged,
        "family": family, "prior_sd": ps, "n": n, "p": p,
        "method": "Bayesian GLM: penalised IRLS to the posterior mode with a "
                  "normal prior, Laplace covariance (Gelman et al. BDA3 "
                  "Ch. 16, Sec. 4.1)",
        "note": "the prior is what separates this from ML -- under "
                "separation the ML coefficient diverges while the posterior "
                "mode stays finite; the Laplace approximation is exact for "
                "the Gaussian family",
    })


def cheatsheet():
    return ("glmbay: bayesian_glm(X, y, family, prior_sd) -> posterior mode, "
            "Laplace covariance and log marginal likelihood (Gelman et al. "
            "2013, Bayesian Data Analysis 3rd ed., Ch. 16)")


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
bayesianglm = bayesian_glm
