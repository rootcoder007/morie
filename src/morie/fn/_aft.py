# morie.fn -- shared helpers (rootcoder007/morie)
"""Shared accelerated-failure-time machinery.

An AFT model is a linear model on the LOG of survival time:

.. math::
    \\log T_i = x_i^\\top \\beta + \\sigma \\epsilon_i,

so :math:`e^{\\beta_j}` is a **time ratio** -- the multiplicative effect on
survival time -- not a hazard ratio. Positive coefficients mean *longer*
survival, the opposite sign convention to Cox. Mixing the two up is the single
most common misreading of an AFT fit.

The error distribution of :math:`\\epsilon` picks the family: extreme-value
gives Weibull, logistic gives log-logistic, normal gives log-normal. Fitting is
by direct maximisation of the censored log-likelihood, with right-censored
observations contributing the log survivor function rather than the log
density.
"""

from __future__ import annotations

from . import _array_core as np

__all__ = ["aft_fit", "log_dens_surv"]


def log_dens_surv(z, family):
    """Standardised log density and log survivor at ``z`` for each family."""
    if family == "weibull":            # extreme-value (Gumbel, min) errors
        return z - np.exp(np.clip(z, -500, 500)), -np.exp(np.clip(z, -500, 500))
    if family == "loglogistic":        # logistic errors
        zz = np.clip(z, -500, 500)
        return zz - 2.0 * np.logaddexp(0.0, zz), -np.logaddexp(0.0, zz)
    if family == "lognormal":          # normal errors
        from ._stats_core import norm

        return norm.logpdf(z), norm.logsf(z)
    raise ValueError(
        f'family must be "weibull", "loglogistic" or "lognormal", got {family!r}'
    )


def _numeric_cov(fn, theta, rel=1e-5):
    """Inverse of the central-difference Hessian of ``fn`` at ``theta``."""
    k = theta.size
    h = rel * np.maximum(np.abs(theta), 1.0)
    H = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            tp = theta.copy(); tp[i] += h[i]; tp[j] += h[j]
            tm = theta.copy(); tm[i] -= h[i]; tm[j] -= h[j]
            tpm = theta.copy(); tpm[i] += h[i]; tpm[j] -= h[j]
            tmp = theta.copy(); tmp[i] -= h[i]; tmp[j] += h[j]
            H[i, j] = H[j, i] = (fn(tp) - fn(tpm) - fn(tmp) + fn(tm)) / (
                4.0 * h[i] * h[j])
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None


def aft_fit(t, e, X, family="weibull", max_iter=500, tol=1e-6, add_intercept=True):
    """Maximise the censored AFT log-likelihood.

    Returns ``(beta, log_scale, loglik, cov, n_iter, converged)`` with ``beta``
    on the log-time scale.
    """
    from ._sci_core import minimize

    n = t.size
    A = np.column_stack([np.ones(n), X]) if add_intercept else np.asarray(X, dtype=float)
    p = A.shape[1]
    logt = np.log(np.maximum(t, 1e-300))

    def nll(theta):
        b, ls = theta[:p], theta[p]
        sigma = np.exp(np.clip(ls, -20, 20))
        z = (logt - A @ b) / sigma
        ld, lsv = log_dens_surv(z, family)
        return -float(np.sum(np.where(e > 0, ld - np.log(sigma), lsv)))

    start = np.r_[np.linalg.lstsq(A, logt, rcond=None)[0], 0.0]
    res = minimize(nll, start, method="BFGS",
                   options={"maxiter": max_iter, "gtol": tol})
    theta = res.x
    # BFGS's hess_inv is a secant approximation accumulated along the path
    # the optimiser happened to walk, not the curvature at the optimum: it
    # depends on the starting point and on the optimiser itself, so the
    # standard errors it implies are not reproducible across solvers. A
    # central-difference Hessian of the same objective is, and it is what
    # the standard errors should have been derived from all along.
    cov = _numeric_cov(nll, theta)

    # BFGS reports success=False on "precision loss" even at a clean optimum,
    # so judge convergence on the gradient itself. Parameter recovery on
    # simulated data is correct in exactly the cases the flag calls failures.
    grad = np.asarray(getattr(res, "jac", np.full(p + 1, np.nan)), dtype=float)
    gnorm = float(np.max(np.abs(grad))) if np.all(np.isfinite(grad)) else np.inf
    converged = bool(res.success or gnorm < 1e-3 * max(1.0, abs(float(res.fun))))
    return (theta[:p], float(theta[p]), float(-res.fun), cov,
            int(res.nit), converged)
