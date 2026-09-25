# morie.fn -- function file (rootcoder007/morie)
"""
MA(q) model fitting by conditional least squares.

Brockwell and Davis's convention, X_t = Z_t + theta_1 Z_{t-1} + ... +
theta_q Z_{t-q}: the coefficients are fitted by minimising the
conditional sum of squared innovations, started from the invertible
lag-one moment solution.

Category: TimeSeries
"""

from . import _array_core as np
from ._sci_core import minimize


def mafit(y, q=1, method="cml"):
    """Fit MA(q) model.

    Parameters
    ----------
    y : array-like
        Univariate time series, shape (n,).
    q : int, optional
        Order of moving average. Default 1.
    method : str, optional
        "cml" (conditional maximum likelihood, i.e. conditional least
        squares). Default "cml".

    Returns
    -------
    TimeSeriesResult
        Fields: ma_coeff (array), sigma2 (float), acf (array), loglik (float), n (int).
        ``acf`` is the model autocorrelation
        rho(k) = (theta_k + sum_j theta_j theta_{j+k}) / (1 + sum_j theta_j^2).

    Examples
    --------
    The model ACF of X_t = Z_t + 0.5 Z_{t-1} is 0.5 / 1.25 = 0.4 at lag 1:

    >>> r = mafit([0.3, -1.2, 0.8, 0.1, -0.4, 1.5, -0.7, 0.2, 0.9, -1.1], q=1)
    >>> th = float(r.ma_coeff[0])
    >>> abs(float(r.acf[1]) - th / (1 + th * th)) < 1e-15
    True

    References
    ----------
    Brockwell, P. J., & Davis, R. A. (2016). Introduction to Time Series
    and Forecasting (3rd ed.). Springer. Section 3.4.
    """
    from ._containers import TimeSeriesResult

    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")

    n = len(y)
    if n <= q:
        raise ValueError(f"Need n > q; got n={n}, q={q}")

    y = y - np.mean(y)

    # Compute sample ACF
    acov = np.array([np.mean(y[:-k] * y[k:]) if k > 0 else np.mean(y**2) for k in range(q + 1)])
    rho = acov / acov[0]

    # Start value: the invertible solution of rho(1) = theta / (1 + theta^2)
    # (Brockwell and Davis Sec. 3.4); |rho(1)| >= 1/2 has none, so start
    # on the boundary's side at +-0.5
    theta_init = np.zeros(q)
    r1 = float(rho[1])
    if r1 == 0.0:
        theta_init[0] = 0.0
    elif abs(r1) < 0.5:
        theta_init[0] = (1.0 - (1.0 - 4.0 * r1 * r1) ** 0.5) / (2.0 * r1)
    else:
        theta_init[0] = 0.5 * (1.0 if r1 > 0 else -1.0)

    def cml_loglik(theta, y, q):
        """Conditional log-likelihood for MA(q)."""
        sigma2 = np.mean(y**2)
        eps = np.copy(y)
        for t in range(q, len(y)):
            eps[t] = y[t] - np.sum(theta * eps[t - q : t][::-1])
        return -np.sum(eps[q:] ** 2) / (2 * sigma2)

    if method == "cml":
        result = minimize(lambda th: -cml_loglik(th, y, q), theta_init, method="BFGS")
        theta = result.x
        loglik = -result.fun
    else:
        raise ValueError(f"Unknown method: {method}")

    # Residual variance via backsubstitution
    eps = np.copy(y)
    for t in range(q, len(y)):
        eps[t] = y[t] - np.sum(theta * eps[t - q : t][::-1])
    sigma2 = np.mean(eps[q:] ** 2)

    # ACF of MA(q)
    acf_ma = np.zeros(q + 1)
    acf_ma[0] = 1.0
    denominator = 1 + np.sum(theta**2)
    for k in range(1, q + 1):
        # theta_0 = 1: gamma(k) / sigma^2 = theta_k + sum_{j=1}^{q-k} theta_j theta_{j+k}
        cross = sum(float(theta[j - 1]) * float(theta[j + k - 1]) for j in range(1, q - k + 1))
        acf_ma[k] = (float(theta[k - 1]) + cross) / denominator

    return TimeSeriesResult(
        name=short,
        values=theta.copy(),
        extra={
            "ma_coeff": theta.copy(),
            "sigma2": float(sigma2),
            "acf": acf_ma.copy(),
            "loglik": float(loglik),
            "n": n,
            "q": q,
        },
    )


short = "mafit"
alias = "ma_fitting"
quote = "The whole is greater than the sum of its parts. -- Aristotle"
__all__ = ["mafit"]


def cheatsheet() -> str:
    return "mafit(y, q=1) -> MA(q) fit via conditional likelihood"
