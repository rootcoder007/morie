# morie.fn -- function file (rootcoder007/morie)
"""Bayesian logistic regression by normal approximation."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bayeslogit", "bayes_logistic"]


def bayeslogit(X, y, prior_sd=10.0, iters=50, tol=1e-12):
    """Posterior mode and normal approximation for logistic regression.

    A proper prior is not decoration here: with separated data the
    likelihood alone has no maximum and the coefficients run to
    infinity, and it is the N(0, prior_sd^2) term that keeps the
    Hessian invertible and the answer finite.  ``prior_sd`` large
    approaches the maximum-likelihood fit.

    The normal approximation is centred at the MODE and scaled by the
    inverse Hessian, so it is exactly as good as the posterior is
    symmetric -- which for logistic regression with small counts it is
    not.  ``iterations`` and ``converged`` are returned so a caller can
    tell a converged fit from a truncated one.

    An intercept column is prepended; do not include one in X.

    Formula: log p(beta | y) = sum_i [ y_i x_i'beta - log(1 + e^{x_i'beta}) ]
                               - ||beta||^2 / (2 s^2) + const;
             Newton step beta <- beta + (X'WX + I/s^2)^-1
                                        (X'(y - p) - beta/s^2),
             W = diag(p_i (1 - p_i));  Var(beta) ~= (X'WX + I/s^2)^-1

    Parameters
    ----------
    X : array-like, shape (n, p)
        Predictors, WITHOUT an intercept column.
    y : array-like
        Binary responses in {0, 1}.
    prior_sd : float
        Standard deviation of the independent N(0, s^2) prior on every
        coefficient including the intercept.
    iters : int
        Maximum Newton steps.
    tol : float
        Convergence tolerance on the maximum coefficient change.

    Returns
    -------
    RichResult
        ``estimate`` (the mode, intercept first), ``se``,
        ``log_posterior``, ``iterations``, ``converged``, ``n``, ``p``.

    References
    ----------
    Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013), Bayesian
    Data Analysis, 3rd edition, Section 4.1 (the normal approximation
    to the posterior, centred at the mode with covariance the inverse
    of the negative second-derivative matrix of the log posterior) and
    Chapter 16 on generalized linear models.  Fetched as the full text
    of the book from the author's own copy.
    """
    X = C.mat(X)
    y = C.vec(y)
    n = len(X)
    if len(y) != n:
        raise ValueError("X and y must have the same number of rows")
    if any(v not in (0.0, 1.0) for v in y):
        raise ValueError("y must be binary 0/1")
    s = float(prior_sd)
    if s <= 0:
        raise ValueError("prior_sd must be positive")
    Z = C.cbind1(X)
    p = len(Z[0])
    if n < p:
        raise ValueError("more coefficients than observations")
    b = [0.0] * p
    inv_s2 = 1.0 / (s * s)
    conv = 0.0
    it = 0
    for it in range(1, int(iters) + 1):
        eta = [sum(Z[i][j] * b[j] for j in range(p)) for i in range(n)]
        mu = [1.0 / (1.0 + math.exp(-min(500.0, max(-500.0, e))))
              for e in eta]
        g = [sum(Z[i][j] * (y[i] - mu[i]) for i in range(n))
             - b[j] * inv_s2 for j in range(p)]
        H = [[sum(Z[i][a] * mu[i] * (1.0 - mu[i]) * Z[i][c]
                  for i in range(n)) + (inv_s2 if a == c else 0.0)
              for c in range(p)] for a in range(p)]
        step = C.solvev(H, g)
        b = [b[j] + step[j] for j in range(p)]
        if max(abs(v) for v in step) < tol:
            conv = 1.0
            break
    eta = [sum(Z[i][j] * b[j] for j in range(p)) for i in range(n)]
    mu = [1.0 / (1.0 + math.exp(-min(500.0, max(-500.0, e)))) for e in eta]
    H = [[sum(Z[i][a] * mu[i] * (1.0 - mu[i]) * Z[i][c] for i in range(n))
          + (inv_s2 if a == c else 0.0) for c in range(p)] for a in range(p)]
    V = C.inv(H)
    ll = sum(y[i] * eta[i] - math.log1p(math.exp(min(500.0, eta[i])))
             for i in range(n))
    lp = ll - 0.5 * inv_s2 * sum(v * v for v in b)
    return RichResult(payload={
        "estimate": b, "se": [math.sqrt(V[j][j]) for j in range(p)],
        "log_posterior": lp, "iterations": float(it), "converged": conv,
        "n": float(n), "p": float(p),
        "method": "Logistic regression posterior mode, BDA3 Section 4.1"})


bayes_logistic = bayeslogit


def cheatsheet():
    return "baylog: Newton to the mode with a N(0,s^2) prior; se from inverse Hessian"
