# morie.fn -- function file (rootcoder007/morie)
"""Negative binomial regression."""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as sp_stats
from ._sci_core import minimize_scalar

from ._containers import DescriptiveResult
from ._stats_core import _digamma


def _alpha_score(la, y, mu):
    """d logL / d log(alpha) for NB2 at fixed means, r = 1/alpha."""
    import math
    r = math.exp(-la)
    s = 0.0
    for yi, m in zip(y, mu):
        s += (_digamma(yi + r) - _digamma(r) + math.log(r / (r + m))
              + (m - yi) / (r + m))
    return -r * s


def _polish_alpha(alpha, y, mu):
    """Secant on the alpha score from the golden-section estimate.

    The bounded search stops at an absolute width of about 1e-8 on
    (1e-6, 100), which left alpha off the MLE in the eighth digit.
    """
    import math
    y = [float(v) for v in y]
    mu = [float(v) for v in mu]
    x0 = math.log(alpha)
    x1 = x0 + 1e-4
    f0 = _alpha_score(x0, y, mu)
    f1 = _alpha_score(x1, y, mu)
    for _ in range(50):
        if f1 == f0 or abs(f1) < 1e-13 * len(y):
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if not math.isfinite(x2) or abs(x2 - x1) > 5.0:
            return alpha
        x0, f0 = x1, f1
        x1, f1 = x2, _alpha_score(x2, y, mu)
    return math.exp(x1)


def negbin_regression(
    y_counts: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-11,
) -> DescriptiveResult:
    """NB2 regression via IRLS with MLE for dispersion.

    Parameters
    ----------
    y_counts : (n,) counts
    X : (n, p)

    Returns
    -------
    DescriptiveResult
    """
    y = np.asarray(y_counts, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    k = X_int.shape[1]

    beta = np.zeros(k)
    alpha = 1.0

    for _ in range(max_iter):
        eta = X_int @ beta
        mu = np.exp(np.clip(eta, -20, 20))
        W = mu / (1 + alpha * mu)
        z = eta + (y - mu) / mu
        try:
            beta_new = np.linalg.solve(X_int.T @ (X_int * W[:, None]), X_int.T @ (W * z))
        except np.linalg.LinAlgError:
            break

        mu_new = np.exp(np.clip(X_int @ beta_new, -20, 20))

        def neg_ll(a):
            a = max(a, 1e-6)
            ll = np.sum(sp_stats.nbinom.logpmf(y.astype(int), n=1 / a, p=1 / (1 + a * mu_new)))
            return -ll

        res = minimize_scalar(neg_ll, bounds=(1e-6, 100), method="bounded")
        alpha = _polish_alpha(res.x, y.tolist(), mu_new.tolist())

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu_final = np.exp(np.clip(X_int @ beta, -20, 20))
    ll = float(np.sum(sp_stats.nbinom.logpmf(y.astype(int), n=1 / alpha, p=1 / (1 + alpha * mu_final))))

    coef_names = ["intercept"] + [f"x{j}" for j in range(p)]

    return DescriptiveResult(
        name="negbin",
        value=float(-2 * ll),
        extra={
            "coefficients": dict(zip(coef_names, beta.tolist())),
            "alpha": float(alpha),
            "log_likelihood": float(ll),
            "aic": float(-2 * ll + 2 * (k + 1)),
            "n": n,
            "k": k,
        },
    )


nbreg = negbin_regression


def cheatsheet() -> str:
    return "negbin_regression({}) -> Negative binomial regression."
