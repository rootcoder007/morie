"""Spatio-temporal autoregressive model (Schabenberger & Gotway Ch 9)."""

from . import _array_core as np


def starn(
    data: np.ndarray,
    weights: np.ndarray,
    *,
    order: int = 1,
) -> dict:
    r"""
    Fit a spatio-temporal autoregressive (STAR) model.

    .. math::

        z_t = \\sum_{k=0}^{p} \\phi_k W^k z_{t-1} + \\varepsilon_t

    where :math:`W` is the spatial weights matrix, :math:`W^0 = I` and
    :math:`p` is the spatial lag order: the STAR(1_p) of Pfeifer and
    Deutsch (1980), with the l-th order neighbour matrix taken as W^l.
    The k = 0 term, each site's own previous value, used to be left
    out, which forced its coefficient to zero.

    :param data: Spatio-temporal data (T, n) -- T time steps, n locations.
    :param weights: Spatial weights matrix (n, n), row-standardised.
    :param order: Spatial lag order *p*.
    :return: dict with ``coefficients``, ``residuals``, ``fitted``, ``aic``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Pfeifer, P. E. & Deutsch, S. J. (1980). A three-stage iterative
    procedure for space-time modeling. *Technometrics*, 22(1), 35-47.

    Schabenberger & Gotway (2005), Ch. 9.
    """
    data = np.asarray(data, dtype=float)
    weights = np.asarray(weights, dtype=float)
    T, n = data.shape
    if weights.shape != (n, n):
        raise ValueError(f"weights must be ({n}, {n}), got {weights.shape}.")
    if T < 2:
        raise ValueError("Need at least 2 time steps.")

    y = data[1:].ravel()
    X_parts = []
    for k in range(0, order + 1):
        Wk = np.linalg.matrix_power(weights, k)
        lagged = (Wk @ data[:-1].T).T
        X_parts.append(lagged.reshape(-1, 1) if lagged.ndim == 1 else lagged.reshape(-1, 1))


    X = np.hstack(X_parts)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coeffs
    residuals = y - fitted

    k_params = len(coeffs)
    n_obs = len(y)
    rss = float(np.sum(residuals**2))
    sigma2 = rss / max(n_obs - k_params, 1)
    # Gaussian AIC uses the ML variance rss / n, not the unbiased one
    aic = (n_obs * float(np.log(rss / n_obs)) + 2 * k_params
           if rss > 0 else float("-inf"))

    return {
        "coefficients": coeffs,
        "residuals": residuals.reshape(T - 1, n),
        "fitted": fitted.reshape(T - 1, n),
        "aic": float(aic),
        "sigma2": sigma2,
        "order": order,
        "T": T,
        "n": n,
    }


starn_fn = starn


def cheatsheet() -> str:
    return "starn({}) -> Spatio-temporal autoregressive (STAR) model."
