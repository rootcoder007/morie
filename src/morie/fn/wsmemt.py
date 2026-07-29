# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""EM algorithm for a two-component normal mixture."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_em_algorithm"]


def _norm_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))


def wasserman_em_algorithm(X, theta0, max_iter=200, tol=1e-8):
    """
    EM for a two-component univariate normal mixture.

    Formula: E-step computes responsibilities
    gamma_i = pi phi(x; mu2, s2) / ((1-pi) phi(x; mu1, s1) + pi phi(x; mu2, s2));
    M-step maximises Q(theta | theta^(t)) in closed form (weighted
    means/variances). Iterates until the log-likelihood gain drops
    below ``tol``. The log-likelihood is monotone non-decreasing —
    checked every iteration, violation raises.

    Parameters
    ----------
    X : array-like
        Sample (n >= 2).
    theta0 : sequence of 5 floats
        (pi, mu1, mu2, sd1, sd2) initial values, 0 < pi < 1, sds > 0.
    max_iter : int
        Iteration cap.
    tol : float
        Log-likelihood convergence gain.

    Returns
    -------
    result : dict
        Keys: estimate (final pi), pi, mu1, mu2, sd1, sd2,
        log_likelihood, iterations, converged, n, method.

    References
    ----------
    Wasserman (2004), Ch 9 (EM); Dempster-Laird-Rubin (1977).

    Examples
    --------
    Well-separated clusters recover their means:

    >>> X = [0.0, 0.1, -0.1, 0.05, 10.0, 10.1, 9.9, 10.05]
    >>> out = wasserman_em_algorithm(X, (0.5, -1.0, 11.0, 1.0, 1.0))
    >>> round(out["pi"], 6)
    0.5
    >>> abs(out["mu1"] - 0.0125) < 1e-6
    True
    >>> abs(out["mu2"] - 10.0125) < 1e-6
    True
    >>> out["converged"]
    True
    >>> wasserman_em_algorithm(X, (1.5, 0, 1, 1, 1))
    Traceback (most recent call last):
        ...
    ValueError: the mixing weight must lie in (0, 1); got 1.5.
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = X.size
    if n < 2:
        raise ValueError("EM on fewer than 2 points is undefined.")
    pi, mu1, mu2, sd1, sd2 = (float(v) for v in theta0)
    if not 0 < pi < 1:
        raise ValueError(f"the mixing weight must lie in (0, 1); got {pi}.")
    if sd1 <= 0 or sd2 <= 0:
        raise ValueError("initial standard deviations must be positive.")
    ll_old = -np.inf
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        d1 = (1.0 - pi) * _norm_pdf(X, mu1, sd1)
        d2 = pi * _norm_pdf(X, mu2, sd2)
        tot = d1 + d2
        tot = np.where(tot > 0, tot, np.finfo(float).tiny)
        ll = float(np.sum(np.log(tot)))
        if ll < ll_old - 1e-10:
            raise RuntimeError(f"EM log-likelihood decreased ({ll_old} -> {ll}); numerical fault.")
        gamma = d2 / tot
        if abs(ll - ll_old) < tol:
            converged = True
            break
        ll_old = ll
        w2 = float(np.sum(gamma)); w1 = n - w2
        pi = w2 / n
        mu1 = float(np.sum((1 - gamma) * X) / w1)
        mu2 = float(np.sum(gamma * X) / w2)
        sd1 = float(np.sqrt(np.sum((1 - gamma) * (X - mu1) ** 2) / w1))
        sd2 = float(np.sqrt(np.sum(gamma * (X - mu2) ** 2) / w2))
        sd1 = max(sd1, 1e-12); sd2 = max(sd2, 1e-12)
    return RichResult(payload={
        "estimate": float(pi), "pi": float(pi),
        "mu1": mu1, "mu2": mu2, "sd1": sd1, "sd2": sd2,
        "log_likelihood": float(ll), "iterations": int(it),
        "converged": bool(converged), "n": int(n),
        "method": "EM 2-component normal mixture, closed-form M-step"})


def cheatsheet():
    return "wsmemt: E-step responsibilities, M-step weighted moments; ll monotone checked"
