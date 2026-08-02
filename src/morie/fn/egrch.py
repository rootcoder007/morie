# morie.fn -- function file (rootcoder007/morie)
"""EGARCH(1,1) asymmetric volatility model.

Native pure-NumPy Gaussian quasi-maximum-likelihood fit; no GARCH package
is imported.
"""

from __future__ import annotations

from . import _array_core as np
from ._sci_core import optimize

from ._richresult import RichResult

__all__ = ["egarch_model"]


def egarch_model(x):
    r"""Fit an EGARCH(1,1) model to financial returns.

    .. math::

        \log(\sigma_t^2) = \omega + \alpha(|z_{t-1}|-E|z_{t-1}|)
        + \gamma z_{t-1} + \beta \log(\sigma_{t-1}^2)

    ``alpha`` is the size effect and ``gamma`` the sign (leverage) effect,
    matching the ``arch`` package's ordering. This naming is not universal --
    ``rugarch`` swaps the two letters -- so compare coefficients by the term
    they multiply, not by name. ``theta`` is kept as an alias for ``gamma``
    because the sign effect is written :math:`\theta` in some presentations.

    Parameters
    ----------
    x : array-like
        Return series (recommended demeaned).

    Returns
    -------
    RichResult
        keys: ``omega``, ``alpha``, ``beta``, ``gamma``, ``theta``,
        ``loglik``, ``n``, ``conditional_variance``, ``method``.

    References
    ----------
    Nelson DB (1991). Conditional Heteroskedasticity in Asset Returns: A
    New Approach. *Econometrica* 59(2), 347-370.
    """
    r = np.asarray(x, dtype=float).ravel()
    r = r - r.mean()
    n = r.size
    if n < 20:
        raise ValueError(f"Need at least 20 observations, got {n}.")


    # Gaussian QMLE ----------------------------------------------------------
    EZ = np.sqrt(2.0 / np.pi)  # E|Z| for standard normal

    def neg_ll(p):
        omega, alpha, gamma, beta = p
        if abs(beta) >= 1.0:
            return 1e10
        log_s2 = np.zeros(n)
        log_s2[0] = np.log(np.var(r) + 1e-12)
        for t in range(1, n):
            z = r[t - 1] / np.sqrt(np.exp(log_s2[t - 1]) + 1e-12)
            log_s2[t] = omega + beta * log_s2[t - 1] + alpha * (np.abs(z) - EZ) + gamma * z
            # Keep exp(log_s2) inside [1e-30, 1e30]. Far wider than any real
            # variance, but it stops the optimiser's gradient probing from
            # overflowing to inf and turning the whole objective into nan --
            # which silently returns the starting values.
            if not np.isfinite(log_s2[t]):
                log_s2[t] = log_s2[t - 1]
            log_s2[t] = min(70.0, max(-70.0, log_s2[t]))
        s2 = np.exp(log_s2)
        ll = -0.5 * np.sum(np.log(2 * np.pi * s2) + r**2 / s2)
        if not np.isfinite(ll):
            return 1e10
        return -ll

    x0 = [0.0, 0.1, 0.0, 0.9]
    bnds = [(-5, 5), (-1.0, 1.0), (-1.0, 1.0), (-0.999, 0.999)]
    # L-BFGS-B is the wrong tool here: the gradient at the start point is
    # O(1e3), its first line search overshoots, backtracks to nothing, and it
    # reports CONVERGENCE after one iteration having moved nowhere. Powell
    # needs no gradient and reaches the optimum in a handful of iterations.
    fit = optimize.minimize(neg_ll, x0, bounds=bnds, method="Powell")
    omega, alpha, gamma, beta = fit.x
    log_s2 = np.zeros(n)
    log_s2[0] = np.log(np.var(r) + 1e-12)
    for t in range(1, n):
        z = r[t - 1] / np.sqrt(np.exp(log_s2[t - 1]) + 1e-12)
        log_s2[t] = omega + beta * log_s2[t - 1] + alpha * (np.abs(z) - EZ) + gamma * z
        if not np.isfinite(log_s2[t]):
            log_s2[t] = log_s2[t - 1]
        log_s2[t] = min(70.0, max(-70.0, log_s2[t]))
    return RichResult(
        payload={
            "omega": float(omega),
            "alpha": float(alpha),
            "gamma": float(gamma),
            "beta": float(beta),
            "theta": float(gamma),  # sign effect, alias of gamma
            "loglik": float(-fit.fun),
            "n": int(n),
            "conditional_variance": np.exp(log_s2),
            "method": "EGARCH(1,1) Gaussian QMLE",
        }
    )


# CANONICAL TEST -------------------------------------------------------------
# rng = np.random.default_rng(0); n=200
# eps = rng.standard_normal(n); s2 = np.zeros(n); r = np.zeros(n)
# omega, alpha, gamma, beta = -0.1, 0.1, -0.05, 0.95
# s2[0] = 1.0
# for t in range(1, n):
#     z = r[t-1]/np.sqrt(s2[t-1]); g = alpha*(abs(z)-np.sqrt(2/np.pi)) + gamma*z
#     s2[t] = np.exp(omega + beta*np.log(s2[t-1]) + g); r[t] = np.sqrt(s2[t])*eps[t]
# egarch_model(r)  # -> beta near 0.95


def cheatsheet():
    return "egrch: EGARCH(1,1) asymmetric volatility (Nelson 1991)."
