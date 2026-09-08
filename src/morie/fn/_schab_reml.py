# SPDX-License-Identifier: AGPL-3.0-or-later
"""Restricted maximum likelihood for the covariance parameters.

Schabenberger & Gotway (2005), Sec. 4.5.2 and Sec. 5.5.3. Three things come
straight from the text rather than from convenience:

1. The matrix of error contrasts K disappears. Sec. 5.5.3 quotes Searle et
   al. (1992, pp. 451-452),

       K'(K Sigma K')^-1 K = Sigma^-1 - Sigma^-1 X Omega X' Sigma^-1,
       Omega = (X' Sigma^-1 X)^-1,

   and, since Omega X' Sigma^-1 Z = beta_hat, Z'K'(K Sigma K')^-1 KZ reduces
   to r' Sigma^-1 r. With Harville's (1974, 1977) choice of K the objective is

       phi_R(theta) = ln|Sigma| + ln|X' Sigma^-1 X| + r' Sigma^-1 r
                      + (n - k) ln(2 pi),

   which never forms K at all. Harville notes that other admissible K differ
   only by a constant free of theta and beta, so the minimiser is the same.

2. A scale parameter is profiled out. Writing Sigma(theta) = sigma^2
   Sigma(theta*), eq (5.49) gives

       sigma^2_reml = r' Sigma(theta*)^-1 r / (n - k)

   and minus twice the profiled restricted log likelihood

       phi_R,sigma(theta*) = ln|Sigma(theta*)| + ln|X' Sigma(theta*)^-1 X|
                             + (n - k) ln(sigma^2_reml)
                             + (n - k)(ln(2 pi) - 1).

   For the models of Sec. 4.3 that leaves TWO free parameters, the nugget
   ratio and the range, in place of three.

3. Sec. 5.5.2 names the optimiser: "successive updates according to a
   nonlinear optimization technique; for example, by way of the
   Newton-Raphson, Quasi-Newton, or some other suitable algorithm". The
   quasi-Newton branch is taken here, driven by the exact gradient below, so
   there is no finite-difference step to choose and both language arms run
   the same arithmetic.

There is no REML estimator of the mean: the text is explicit that
beta_reml is "simply an EGLS estimator evaluated at theta_reml".

References
----------
Schabenberger, O. & Gotway, C. A. (2005) *Statistical Methods for
Spatial Data Analysis*, Texts in Statistical Science, Chapman &
Hall/CRC, Boca Raton, ISBN 1-58488-322-7.
Sec. 4.5 and Sec. 5.5.

Harville, D. A. (1974) "Bayesian inference for variance components
using only error contrasts", *Biometrika* 61, 383-385 -- the origin of
the restricted likelihood maximised here.

Everything here is internal.
"""

from . import _array_core as np

from ._schab_gn import semivariogram_jacobian
from ._schab_vario import correlogram

__all__ = []


def correlation_matrix(coords, nugget_ratio, rng, model):
    """Sigma(theta*) = xi I + (1 - xi) R(h; a), the scale-free structure.

    Factoring sigma^2 = c0 + sigma0^2 out of Sigma(theta) leaves the nugget
    as a RATIO xi = c0 / (c0 + sigma0^2) in [0, 1]. That is the
    reparameterisation Sec. 5.5.2 calls for when it writes
    Sigma(theta) = sigma^2 Sigma(theta*).
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    r = correlogram(d, rng, model)
    sigma = (1.0 - nugget_ratio) * r
    sigma[np.diag_indices_from(sigma)] = 1.0
    return sigma, d, r


def _dsigma(d, r, nugget_ratio, rng, model):
    """dSigma(theta*)/d(xi, a).

    d/dxi is I - R, and d/da is (1 - xi) dR/da; dR/da is the same expression
    the Gauss-Newton Jacobian uses, so the two fitters share one derivation.
    """
    n = d.shape[0]
    d_xi = -r.copy()
    d_xi[np.diag_indices_from(d_xi)] = 0.0
    flat = semivariogram_jacobian(d.ravel(), 0.0, 1.0, rng, model)[:, 2]
    dr_da = (-flat).reshape(n, n)     # jacobian column is -sill * dR/da at sill = 1
    d_a = (1.0 - nugget_ratio) * dr_da
    d_a[np.diag_indices_from(d_a)] = 0.0
    return d_xi, d_a


def profiled_reml(coords, z, X, nugget_ratio, rng, model):
    """eq (5.49) and its exact gradient with respect to (xi, a).

    Returns (value, gradient, sigma2, beta) or (inf, zeros, nan, nan) when
    theta* leaves the positive-definite region.
    """
    n, k = X.shape
    if not (0.0 <= nugget_ratio <= 1.0) or rng <= 0.0:
        return np.inf, np.zeros(2), np.nan, np.full(k, np.nan)
    sigma, d, r_corr = correlation_matrix(coords, nugget_ratio, rng, model)
    try:
        chol = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        return np.inf, np.zeros(2), np.nan, np.full(k, np.nan)
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    sinv = np.linalg.inv(sigma)
    sx = sinv @ X
    xsx = X.T @ sx
    try:
        omega = np.linalg.inv(xsx)
    except np.linalg.LinAlgError:
        return np.inf, np.zeros(2), np.nan, np.full(k, np.nan)
    sign, logdet_xsx = np.linalg.slogdet(xsx)
    if sign <= 0:
        return np.inf, np.zeros(2), np.nan, np.full(k, np.nan)
    beta = omega @ (sx.T @ z)
    resid = z - X @ beta
    sr = sinv @ resid
    rss = float(resid @ sr)
    dof = n - k
    sigma2 = rss / dof
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return np.inf, np.zeros(2), np.nan, np.full(k, np.nan)
    value = (logdet + logdet_xsx + dof * np.log(sigma2)
             + dof * (np.log(2.0 * np.pi) - 1.0))

    # Exact gradient. For each derivative D of Sigma(theta*):
    #   d ln|Sigma|            = tr(Sigma^-1 D)
    #   d ln|X'Sigma^-1 X|     = -tr(Omega X'Sigma^-1 D Sigma^-1 X)
    #   d (n-k) ln(sigma^2)    = -r'Sigma^-1 D Sigma^-1 r / sigma^2
    # The terms in d beta / d theta* cancel through the GLS normal equations.
    grad = np.zeros(2)
    for j, dmat in enumerate(_dsigma(d, r_corr, nugget_ratio, rng, model)):
        sd = sinv @ dmat
        t1 = float(np.trace(sd))
        t2 = -float(np.trace(omega @ (sx.T @ dmat @ sx)))
        t3 = -float(sr @ dmat @ sr) / sigma2
        grad[j] = t1 + t2 + t3
    return float(value), grad, float(sigma2), beta


def fit_reml(coords, z, X, model="exponential", start=(0.1, None),
             max_iter=200, tol=1e-10):
    """Quasi-Newton (BFGS) on eq (5.49), driven by the exact gradient.

    Sec. 5.5.2 sanctions "Newton-Raphson, Quasi-Newton, or some other
    suitable algorithm"; BFGS needs only the gradient, which is analytic
    here, so no finite-difference step enters and the two language arms
    execute the same steps. The line search is Armijo backtracking with the
    textbook constants c1 = 1e-4 and rho = 1/2 (Nocedal & Wright), fixed
    rather than tuned.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = z.size
    if coords.shape[0] != n or X.shape[0] != n:
        raise ValueError("`coords`, `z` and `X` must agree on the sample size")
    if X.shape[1] >= n:
        raise ValueError("design matrix leaves no error contrasts")

    xi0 = float(start[0])
    a0 = float(start[1]) if start[1] is not None else _default_range(coords)
    x = np.array([_logit(xi0), np.log(a0)])

    def wrapped(u):
        """(5.49) on an unconstrained scale: xi = logistic(u0) keeps the
        nugget ratio in [0, 1] and a = exp(u1) keeps the range positive, so
        the constraints of Sec. 4.3 hold by construction rather than by
        clipping."""
        xi = _logistic(u[0])
        a = np.exp(u[1])
        val, g, s2, beta = profiled_reml(coords, z, X, xi, a, model)
        if not np.isfinite(val):
            return np.inf, np.zeros(2), s2, beta
        chain = np.array([xi * (1.0 - xi), a])
        return val, g * chain, s2, beta

    val, grad, sigma2, beta = wrapped(x)
    hess_inv = np.eye(2)
    for _ in range(max_iter):
        if not np.isfinite(val) or np.max(np.abs(grad)) < tol:
            break
        direction = -hess_inv @ grad
        slope = float(grad @ direction)
        if slope >= 0:                       # not a descent direction; reset
            hess_inv = np.eye(2)
            direction = -grad
            slope = float(grad @ direction)
        step = 1.0
        moved = False
        for _ in range(60):
            trial = x + step * direction
            t_val, t_grad, t_s2, t_beta = wrapped(trial)
            if np.isfinite(t_val) and t_val <= val + 1e-4 * step * slope:
                moved = True
                break
            step *= 0.5
        if not moved:
            break
        s = trial - x
        y = t_grad - grad
        sy = float(s @ y)
        if sy > 1e-300:                       # BFGS update, skipped if unstable
            rho = 1.0 / sy
            eye = np.eye(2)
            hess_inv = ((eye - rho * np.outer(s, y)) @ hess_inv
                        @ (eye - rho * np.outer(y, s)) + rho * np.outer(s, s))
        x, val, grad, sigma2, beta = trial, t_val, t_grad, t_s2, t_beta

    xi = _logistic(x[0])
    a = float(np.exp(x[1]))
    return {"nugget_ratio": float(xi), "range": a, "sigma2": float(sigma2),
            "nugget": float(xi * sigma2), "partial_sill": float((1.0 - xi) * sigma2),
            "beta": beta, "neg2_restricted_loglik": float(val),
            "converged": bool(np.max(np.abs(grad)) < 1e-6)}


def _default_range(coords):
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return max(float(d.max()) / 4.0, 1e-6)


def _logistic(u):
    return 1.0 / (1.0 + np.exp(-u))


def _logit(p):
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return float(np.log(p / (1.0 - p)))
