# morie.fn -- function file (rootcoder007/morie)
"""Tweedie regression (compound Poisson-gamma GLM)."""

import math

from . import _array_core as np
from ._sci_core import minimize
from ._stats_core import norm

from morie.fn._containers import RegressionResult


def rey_tw(
    df, y: str = "y", x: list | str = "x", power: float = 1.5, alpha: float = 0.05, cdf=None
) -> RegressionResult:
    r"""
    Tweedie regression via quasi-likelihood maximisation.

    The Tweedie family unifies Poisson (*p* = 1), compound Poisson-gamma
    (1 < *p* < 2), and gamma (*p* = 2) distributions under a single
    variance function :math:`V(\\mu) = \\mu^p`.

    :param df: DataFrame with response and predictor columns.
    :param y: Response column name (non-negative).
    :param x: Predictor column name(s).
    :param power: Tweedie power parameter. Must be in (1, 2). Default 1.5.
    :param alpha: Significance level. Default 0.05.
    :return: :class:`RegressionResult` with Tweedie coefficients.
    :raises ValueError: On invalid power or negative response.

    References
    ----------
    Jorgensen, B. (1987). Exponential dispersion models. *Journal of
    the Royal Statistical Society: Series B*, 49(2), 127-162.

    Tweedie, M. C. K. (1984). An index which distinguishes between
    some important exponential families. In J. K. Ghosh & J. Roy (Eds.),
    *Statistics: Applications and New Directions*, pp. 579-604.
    """

    if isinstance(x, str):
        x = [x]
    for col in [y] + x:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in DataFrame.")
    if not (1.0 < power < 2.0):
        raise ValueError(f"power must be in (1, 2), got {power}.")

    y_arr = np.asarray(df[y], dtype=float)
    X_arr = np.column_stack([np.ones(len(df))] + [np.asarray(df[c], dtype=float) for c in x])
    n, p_dim = X_arr.shape

    if np.any(y_arr < 0):
        raise ValueError("Response must be non-negative for Tweedie regression.")

    pw = power

    # Fisher scoring on the quasi-score, the standard GLM fit (McCullagh
    # and Nelder 1989, ch. 2): with the log link and V(mu) = mu^p the
    # working weight is mu^(2 - p) and the working response
    # eta + (y - mu) / mu. It replaced a BFGS minimisation at default
    # tolerance (coefficients good to about 1e-8) whose standard errors
    # were BFGS's approximate inverse Hessian with no dispersion factor.
    Xl = X_arr.tolist()
    yl = [float(v) for v in y_arr.tolist()]
    beta_l = [float(v) for v in np.linalg.lstsq(
        X_arr, np.log(np.maximum(y_arr, 0.5)), rcond=None)[0].tolist()]

    def _wls(beta):
        eta = [sum(b * xv for b, xv in zip(beta, r)) for r in Xl]
        mu = [math.exp(e) for e in eta]
        w = [m ** (2.0 - pw) for m in mu]
        z = [e + (yv - m) / m for e, yv, m in zip(eta, yl, mu)]
        XtWX = [[sum(w[i] * Xl[i][a] * Xl[i][c] for i in range(n)) for c in range(p_dim)]
                for a in range(p_dim)]
        XtWz = [sum(w[i] * Xl[i][a] * z[i] for i in range(n)) for a in range(p_dim)]
        return XtWX, XtWz

    for _ in range(100):
        A, bvec = _wls(beta_l)
        new_b = [float(v) for v in np.linalg.solve(np.array(A), np.array(bvec)).tolist()]
        step = max(abs(a - b) for a, b in zip(new_b, beta_l))
        beta_l = new_b
        if step < 1e-13 * (1.0 + max(abs(v) for v in beta_l)):
            break
    beta_hat = np.array(beta_l)
    mu_hat = np.exp(X_arr @ beta_hat)
    residuals = y_arr - mu_hat

    # Dispersion: Pearson chi-square over residual degrees of freedom
    V_mu = mu_hat**pw
    pearson_resid = (y_arr - mu_hat) / np.sqrt(V_mu)
    phi = np.sum(pearson_resid**2) / max(n - p_dim, 1)

    # Var(beta) = phi (X'WX)^-1 at the fit
    A, _ = _wls(beta_l)
    cov = np.linalg.inv(np.array(A)) * phi
    se_beta = np.sqrt(np.maximum(np.diag(cov), 0.0))

    def neg_quasi_ll(beta):
        mu = np.exp(X_arr @ beta)
        if pw == 1:
            return -np.sum(y_arr * np.log(mu) - mu)
        if pw == 2:
            return -np.sum(-y_arr / mu - np.log(mu))
        return -np.sum(y_arr * mu ** (1 - pw) / (1 - pw) - mu ** (2 - pw) / (2 - pw))

    class _R:
        fun = float(neg_quasi_ll(beta_hat))
    result = _R()

    z_vals = beta_hat / np.where(se_beta > 0, se_beta, np.inf)
    p_vals = 2.0 * (norm.sf(np.abs(z_vals)))

    # Tweedie deviance
    dev = 2.0 * np.sum(
        # y^(2-p) is exactly 0 at y = 0 for 1 < p < 2; flooring y at 1e-10
        # added 2e-5 / ((1-p)(2-p)) per zero to the deviance
        y_arr ** (2 - pw) / ((1 - pw) * (2 - pw))
        - y_arr * mu_hat ** (1 - pw) / (1 - pw)
        + mu_hat ** (2 - pw) / (2 - pw)
    )
    aic = 2.0 * p_dim - 2.0 * (-result.fun)

    names = ["intercept"] + list(x)

    return RegressionResult(
        method="Tweedie GLM",
        coefficients=dict(zip(names, beta_hat.tolist())),
        se=dict(zip(names, se_beta.tolist())),
        p_values=dict(zip(names, p_vals.tolist())),
        residuals=residuals,
        fitted=mu_hat,
        n=n,
        k=p_dim,
        extra={"aic": float(aic), "power": power, "phi": float(phi), "deviance": float(dev)},
    )


def cheatsheet() -> str:
    return "rey_tw({}) -> Tweedie regression (compound Poisson-gamma GLM)."


# compact alias per ledger/NAMING.md
reytw = rey_tw
