# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet-multinomial conjugate model."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["dirichlet_multinomial"]


def dirichlet_multinomial(counts, alpha=1.0):
    """
    Dirichlet-multinomial conjugate model

    Formula: p ~ Dir(alpha); y ~ Multinomial(n, p)

    The posterior is Dir(alpha + y), so the posterior mean of p_j is
    (y_j + alpha_j) / (n + sum alpha) and the marginal likelihood is the
    Polya distribution
    n! Gamma(A)/Gamma(n+A) prod Gamma(y_j+a_j)/(y_j! Gamma(a_j)).  With
    two categories this is exactly the beta-binomial.

    Parameters
    ----------
    counts : array-like
        Observed category counts, non-negative.
    alpha : float or array-like
        Dirichlet prior; a scalar is recycled over the categories.

    Returns
    -------
    result : dict
        Keys: estimate (posterior mean vector), post_mean, post_var,
        post_alpha, log_marginal, n, K.

    References
    ----------
    Gelman et al. (2013), Bayesian Data Analysis, 3rd ed., CRC, ch. 3.
    """
    y = core.vec(counts)
    K = len(y)
    if K == 0:
        raise ValueError("empty input: counts has no categories")
    if any(v < 0.0 for v in y):
        raise ValueError("counts must be non-negative")
    a = core.vec(alpha)
    if len(a) == 1:
        a = a * K
    if len(a) != K:
        raise ValueError("alpha must be scalar or one value per category")
    if any(v <= 0.0 for v in a):
        raise ValueError("alpha must be strictly positive")
    n = sum(y)
    A = sum(a)
    post = [a[j] + y[j] for j in range(K)]
    P = A + n
    mean = [post[j] / P for j in range(K)]
    var = [post[j] * (P - post[j]) / (P * P * (P + 1.0)) for j in range(K)]
    lm = math.lgamma(n + 1.0) + math.lgamma(A) - math.lgamma(n + A)
    for j in range(K):
        lm += math.lgamma(y[j] + a[j]) - math.lgamma(y[j] + 1.0) \
            - math.lgamma(a[j])
    return RichResult(payload={
        "estimate": mean[0],
        "post_mean": mean,
        "post_var": var,
        "post_alpha": post,
        "log_marginal": lm,
        "n": n,
        "K": K,
        "method": "Dirichlet-multinomial conjugate model",
    })


def cheatsheet():
    return "diripr: Dirichlet-multinomial conjugate model"


# compact alias per ledger/NAMING.md
dirichletmultinomial = dirichlet_multinomial
