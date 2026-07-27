# morie.fn -- function file (rootcoder007/morie)
"""Quasi-Kalman filter for the log-variance SV(1) model."""

import numpy as np

from ._richresult import RichResult

__all__ = ["vol_stochastic_kalman"]


def vol_stochastic_kalman(r, phi=0.95, sigma_eta=0.2, mu=None):
    r"""Harvey-Ruiz-Shephard QML filter for stochastic volatility.

    Linearise the SV model by squaring and logging the returns:

    .. math:: \log r_t^2 = h_t + \log z_t^2, \qquad
              h_{t+1} = \mu(1-\phi) + \phi h_t + \eta_t,

    then treat :math:`\log z_t^2` as Gaussian with mean
    :math:`E[\log \chi^2_1] = -1.2704` and variance :math:`\pi^2/2`
    and run the ordinary Kalman filter -- quasi-ML because the
    measurement error is actually log-chi-squared, which the docstring
    says instead of pretending exactness.

    Parameters
    ----------
    r : array-like, shape (n,)
        Return series (zeros are floored to avoid log 0).
    phi : float in (-1, 1), default 0.95
        Log-variance persistence.
    sigma_eta : float > 0, default 0.2
        Std dev of the log-variance innovation.
    mu : float, optional
        Unconditional mean of h; default from the sample.

    Returns
    -------
    RichResult
        keys: ``h_filtered`` (n,), ``sigma`` (exp(h/2)), ``loglik``
        (quasi), ``phi``, ``sigma_eta``, ``n``, ``method``.

    References
    ----------
    Harvey, A., Ruiz, E. & Shephard, N. (1994). Multivariate
    stochastic variance models. *The Review of Economic Studies*,
    61(2), 247-264. (the QML linearisation)
    """
    r = np.asarray(r, dtype=float).ravel()
    n = r.size
    if n < 10:
        raise ValueError(f"need at least 10 returns, got {n}.")
    phi = float(phi)
    if not -1 < phi < 1:
        raise ValueError(f"phi must lie in (-1, 1), got {phi}.")
    se = float(sigma_eta)
    if se <= 0:
        raise ValueError(f"sigma_eta must be positive, got {se}.")

    y = np.log(np.maximum(r**2, 1e-12))
    c_mean = -1.2704  # E[log chi2_1]
    c_var = np.pi**2 / 2.0
    if mu is None:
        mu = float(y.mean() - c_mean)
    mu = float(mu)

    h = mu
    P = se**2 / (1 - phi**2)
    hs = np.empty(n)
    ll = 0.0
    for t in range(n):
        # predict
        h_pred = mu * (1 - phi) + phi * h
        P_pred = phi**2 * P + se**2
        # update with y_t = h_t + noise(c_mean, c_var)
        F = P_pred + c_var
        v = y[t] - (h_pred + c_mean)
        K = P_pred / F
        h = h_pred + K * v
        P = P_pred * (1 - K)
        hs[t] = h
        ll += -0.5 * (np.log(2 * np.pi * F) + v**2 / F)

    return RichResult(
        payload={
            "h_filtered": hs,
            "sigma": np.exp(hs / 2.0),
            "loglik": float(ll),
            "phi": phi,
            "sigma_eta": se,
            "n": int(n),
            "method": "SV(1) quasi-Kalman filter (Harvey-Ruiz-Shephard linearisation)",
        }
    )


def cheatsheet():
    return "volsk: Kalman on log r^2 = h + log z^2, noise mean -1.2704 var pi^2/2"
