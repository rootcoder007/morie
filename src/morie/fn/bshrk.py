# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Bayesian shrinkage (horseshoe prior)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_horseshoe(
    X: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    *,
    n_iter: int = 3000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Bayesian linear regression with horseshoe prior via Gibbs sampling.

    Prior (Makalic & Schmidt 2016, Eqs. 1-2): beta_j | lambda_j, tau,
    sigma ~ N(0, lambda_j^2 * tau^2 * sigma^2), lambda_j ~ C+(0, 1),
    tau ~ C+(0, 1), p(sigma^2) ~ 1/sigma^2 -- note sigma^2 INSIDE the
    prior variance; all conditionals below are their Eqs. (9)-(11).

    :param X: Design matrix (n, p).
    :param y: Response vector (n,).
    :param n_iter: Number of Gibbs iterations.
    :param seed: Random seed.
    :return: Dictionary with beta_samples, tau_samples, posterior_mean.

    References
    ----------
    Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe
    estimator for sparse signals. *Biometrika*, 97(2), 465-480.
    Makalic, E., & Schmidt, D. F. (2016). A simple sampler for the
    horseshoe estimator. *IEEE Signal Processing Letters*, 23(1),
    179-182. (arXiv:1508.03884; the sampler implemented here.)
    """
    rng = np.random.default_rng(seed)
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n, p = X_arr.shape

    beta = np.zeros(p)
    sigma2 = 1.0
    tau2 = 1.0
    lambda2 = np.ones(p)
    nu = np.ones(p)
    xi = 1.0

    beta_samples = np.empty((n_iter, p))
    tau_samples = np.empty(n_iter)

    XtX = X_arr.T @ X_arr
    Xty = X_arr.T @ y_arr

    for it in range(n_iter):
        # Eq. (9): A = X'X + Lambda_*^{-1}, beta ~ N(A^{-1}X'y, sigma^2 A^{-1}).
        # The previous code mixed parameterisations -- prior precision
        # WITHOUT sigma^2 here but WITH it in the sigma^2/lambda/tau
        # rates -- so no single model had these as its conditionals.
        A = XtX + np.diag(1.0 / (lambda2 * tau2 + 1e-30))
        A_inv = np.linalg.inv(A)
        beta = rng.multivariate_normal(A_inv @ Xty, sigma2 * A_inv)

        resid = y_arr - X_arr @ beta
        post_a = (n + p) / 2.0  # Eq. (10); the former +1 matched no prior
        post_b = 0.5 * float(resid @ resid) + 0.5 * float(np.sum(beta**2 / (lambda2 * tau2 + 1e-30)))
        sigma2 = 1.0 / rng.gamma(post_a, 1.0 / post_b)

        for j in range(p):
            rate = beta[j] ** 2 / (2 * sigma2 * tau2 + 1e-30) + 1.0 / (nu[j] + 1e-30)
            lambda2[j] = 1.0 / rng.gamma(1.0, 1.0 / (rate + 1e-30))

        rate_tau = np.sum(beta**2 / (lambda2 * sigma2 + 1e-30)) / 2.0 + 1.0 / (xi + 1e-30)
        tau2 = 1.0 / rng.gamma((p + 1) / 2.0, 1.0 / (rate_tau + 1e-30))

        for j in range(p):
            nu[j] = 1.0 / rng.gamma(1.0, 1.0 / (1.0 + 1.0 / (lambda2[j] + 1e-30)))
        xi = 1.0 / rng.gamma(1.0, 1.0 / (1.0 + 1.0 / (tau2 + 1e-30)))

        beta_samples[it] = beta
        tau_samples[it] = np.sqrt(tau2)

    return {
        "beta_samples": beta_samples,
        "tau_samples": tau_samples,
        "posterior_mean": np.mean(beta_samples, axis=0).tolist(),
        "posterior_sd": np.std(beta_samples, axis=0, ddof=1).tolist(),
        "n_iter": n_iter,
    }


bshrk = bayesian_horseshoe


def cheatsheet() -> str:
    return "bayesian_horseshoe({}) -> Bayesian shrinkage (horseshoe prior)."
