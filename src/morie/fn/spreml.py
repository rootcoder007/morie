# SPDX-License-Identifier: AGPL-3.0-or-later
"""Restricted maximum likelihood (REML) for semivariogram parameters."""

import numpy as np
from scipy.optimize import minimize

from ._richresult import RichResult
from ._schab_fit import (_start_and_bounds, covariance_matrix,
                         error_contrasts)
from ._schab_vario import empirical_semivariogram

__all__ = ["schabenberger_reml_variogram"]


def schabenberger_reml_variogram(coords, z, X=None, variogram_model="exponential"):
    """Estimate covariance parameters by restricted maximum likelihood.

    Minimises minus twice the restricted log likelihood, eq (4.39),

        phi_R(theta) = ln|K Sigma(theta) K'| + (n - p) ln(2 pi)
                       + Z' K' (K Sigma(theta) K')^{-1} K Z,

    where K is a matrix of error contrasts, chosen so that E[K Z(s)] = 0.
    REML maximises the likelihood of K Z(s) rather than of Z(s), which is
    what removes the mean from the problem and mitigates the downward bias
    of the ML variance estimates (Patterson and Thompson, 1971).

    K is not unique. Sec. 4.5.2 writes it explicitly for the intercept-only
    mean and notes, citing Harville (1974), that the choice does not affect
    the estimates; here it is taken as an orthonormal basis for the
    orthogonal complement of the column space of ``X``, which works for any
    linear mean structure and reproduces that property.

    There is no REML estimator of the mean. The text is explicit on this:
    the mean is recovered afterwards by evaluating the generalized least
    squares estimator (4.40) at theta_reml, which is what ``mean`` reports.

    Note on the previous docstring of this module, which gave
    ``log L_R = log L + 0.5 log|X' Sigma^{-1} X|``: that sign is wrong --
    the determinant term is subtracted from the log likelihood, equivalently
    added to minus-two-log-likelihood. Working from (4.39) directly avoids
    the question.

    Parameters
    ----------
    coords : array-like, shape (n, d)
        Sampling locations.
    z : array-like, shape (n,)
        Observed values.
    X : array-like, shape (n, p), optional
        Design matrix for the mean. Defaults to an intercept, which is the
        E[Z(s)] = mu case worked in the text.
    variogram_model : {"exponential", "gaussian", "spherical"}
        Which parametric family of Sec. 4.3 to fit.

    Returns
    -------
    RichResult
        Keys: ``nugget``, ``partial_sill``, ``sill``, ``range``, ``mean``,
        ``neg2_restricted_loglik``, ``converged``, ``n``, ``n_contrasts``.

    References
    ----------
    Schabenberger Ch 4, Sec 4.5.2
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    n = z.size
    if coords.shape[0] != n:
        raise ValueError("`coords` and `z` must have the same number of rows")
    X = np.ones((n, 1)) if X is None else np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != n:
        raise ValueError("`X` must have one row per observation")

    K = error_contrasts(X)
    KZ = K @ z

    def neg2_reml(theta):
        nugget, sill, rng = theta
        if nugget < 0 or sill < 0 or rng <= 0 or (nugget + sill) <= 0:
            return np.inf
        sigma = covariance_matrix(coords, nugget, sill, rng, variogram_model)
        m = K @ sigma @ K.T
        try:
            chol = np.linalg.cholesky(m)
        except np.linalg.LinAlgError:
            return np.inf          # theta outside the positive-definite region
        logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
        sol = np.linalg.solve(m, KZ)
        return logdet + K.shape[0] * np.log(2.0 * np.pi) + float(KZ @ sol)

    lags, ghat, counts = _empirical(coords, z)
    start, bounds = _start_and_bounds(lags, ghat)
    # Nelder-Mead rather than L-BFGS-B: phi_R is +inf outside the
    # positive-definite region, so a numerical gradient straddling that
    # boundary comes back non-finite and a quasi-Newton solver quits at the
    # starting point while reporting success. A simplex never differences
    # across the barrier.
    best_x, best_f = np.asarray(start, dtype=float), neg2_reml(start)
    for frac in (0.05, 0.2, 0.5):
        for rscale in (0.5, 1.0, 2.0):
            x0 = np.clip(np.array([frac * start[1], start[1], rscale * start[2]]),
                         [b[0] for b in bounds], [b[1] for b in bounds])
            res = minimize(neg2_reml, x0, method="Nelder-Mead", bounds=bounds,
                           options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-8})
            if np.isfinite(res.fun) and res.fun < best_f:
                best_x, best_f = np.asarray(res.x, dtype=float), float(res.fun)
    nugget, sill, rng = (float(v) for v in best_x)
    best = type("_R", (), {"x": best_x, "fun": best_f,
                           "success": bool(best_f < neg2_reml(start))})()

    # eq (4.40): the EGLS estimator evaluated at theta_reml.
    sigma = covariance_matrix(coords, nugget, sill, rng, variogram_model)
    sinv_x = np.linalg.solve(sigma, X)
    beta = np.linalg.solve(X.T @ sinv_x, sinv_x.T @ z)

    return RichResult(
        title="REML covariance-parameter estimates",
        summary_lines=[("nugget", nugget), ("partial sill", sill),
                       ("range", rng), ("-2 restricted logL", float(best.fun))],
        payload={"nugget": nugget, "partial_sill": sill, "sill": nugget + sill,
                 "range": rng, "mean": beta if beta.size > 1 else float(beta[0]),
                 "neg2_restricted_loglik": float(best.fun),
                 "converged": bool(best.success), "n": int(n),
                 "n_contrasts": int(K.shape[0]), "model": variogram_model,
                 "method": "restricted maximum likelihood"},
    )


def _empirical(coords, z):
    """Starting values only -- REML itself avoids the binning entirely."""
    lag, gam, cnt = empirical_semivariogram(coords, z)
    return (np.asarray(lag, dtype=float), np.asarray(gam, dtype=float),
            np.asarray(cnt, dtype=float))


def cheatsheet():
    return "spreml: REML covariance-parameter estimates (Schabenberger Sec 4.5.2)"
