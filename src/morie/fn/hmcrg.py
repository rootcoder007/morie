# morie.fn -- function file (rootcoder007/morie)
"""Conjugate hierarchical normal model at a given population sd."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["hiermodel", "hierarchical_model"]


def hiermodel(y, sigma, tau):
    """Hierarchical normal model: shrinkage of group means towards mu.

    The shrinkage factor is the whole story: each group mean is pulled
    towards the population mean by an amount set by the RATIO of its
    own noise to the between-group spread.  tau -> 0 gives complete
    pooling (every theta equals mu), tau -> infinity gives none (every
    theta equals its own y).  Both limits are returned as the
    shrinkage vector so the reader can see which regime the data sit
    in.

    tau is taken as GIVEN; the log marginal posterior of tau, up to an
    additive constant and under a flat prior on tau, is returned so a
    caller can profile or grid over it.

    Formula: mu_hat = sum_j y_j/(sigma_j^2 + tau^2) / sum_j 1/(sigma_j^2 + tau^2),
             V_mu^-1 = sum_j 1/(sigma_j^2 + tau^2)                    (5.20)
             theta_hat_j = (y_j/sigma_j^2 + mu/tau^2)/(1/sigma_j^2 + 1/tau^2),
             V_j = 1/(1/sigma_j^2 + 1/tau^2)                          (5.17)
             log p(tau|y) = 0.5 log V_mu
                            - 0.5 sum_j [ log(sigma_j^2 + tau^2)
                                          + (y_j - mu_hat)^2/(sigma_j^2 + tau^2) ]  (5.21)

    Parameters
    ----------
    y : array-like
        Group means y_.j.
    sigma : array-like
        Known within-group standard errors, strictly positive.
    tau : float
        Between-group standard deviation, tau >= 0.

    Returns
    -------
    RichResult
        ``mu_hat``, ``V_mu``, ``theta_hat``, ``V_theta``,
        ``shrinkage``, ``log_post_tau``, ``tau``, ``J``.

    References
    ----------
    Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013), Bayesian
    Data Analysis, 3rd edition, Section 5.4, equations (5.17), (5.20)
    and (5.21).  Fetched as the full text of the book from the author's
    own copy.  Equation (5.21) is given there up to the prior p(tau);
    a flat prior is used here and the additive constant is dropped, so
    the value is comparable across tau but not an absolute density.
    """
    y = C.vec(y)
    s = C.vec(sigma)
    J = len(y)
    if len(s) != J:
        raise ValueError("y and sigma must have the same length")
    if J < 2:
        raise ValueError("a hierarchical model needs at least two groups")
    if any(v <= 0 for v in s):
        raise ValueError("the within-group standard errors must be positive")
    tau = float(tau)
    if tau < 0:
        raise ValueError("tau must be non-negative")
    w = [1.0 / (s[j] ** 2 + tau ** 2) for j in range(J)]
    Vmu = 1.0 / sum(w)
    mu = sum(w[j] * y[j] for j in range(J)) * Vmu
    if tau == 0.0:
        th = [mu] * J
        Vt = [0.0] * J
        shr = [1.0] * J
    else:
        prec = [1.0 / s[j] ** 2 + 1.0 / tau ** 2 for j in range(J)]
        Vt = [1.0 / p for p in prec]
        th = [(y[j] / s[j] ** 2 + mu / tau ** 2) / prec[j] for j in range(J)]
        shr = [(1.0 / s[j] ** 2) / prec[j] for j in range(J)]
        shr = [1.0 - v for v in shr]
    lp = 0.5 * math.log(Vmu) - 0.5 * sum(
        math.log(s[j] ** 2 + tau ** 2) + (y[j] - mu) ** 2 * w[j]
        for j in range(J))
    return RichResult(payload={
        "mu_hat": mu, "V_mu": Vmu, "theta_hat": th, "V_theta": Vt,
        "shrinkage": shr, "log_post_tau": lp, "tau": tau, "J": float(J),
        "method": "Hierarchical normal model, BDA3 (5.17)/(5.20)/(5.21)"})


hierarchical_model = hiermodel


def cheatsheet():
    return "hmcrg: theta_j = precision-weighted (y_j, mu); shrinkage = sigma-driven"
