# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quantile (pinball) loss for point forecasts."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["qrF", "quantile_forecast"]


def qrF(y, y_hat, tau):
    """
    Quantile (pinball) scoring function for point forecasts.

    For forecast x, realization y and level tau in (0, 1),

        S(x, y) = (1{x >= y} - tau) (x - y),

    the generalized piecewise-linear scoring function of order tau
    with g(x) = x (Gneiting 2011, eq. for GPL scoring functions,
    Theorem 9). S is negatively oriented (smaller is better); the
    tau-quantile of the predictive distribution is the optimal point
    forecast under S. Equivalent to the Koenker-Bassett check
    function rho_tau(u) = u (tau - 1{u < 0}) with u = y - x.

    Parameters
    ----------
    y : array-like
        Realizations.
    y_hat : array-like
        Forecasts (broadcast against y).
    tau : float
        Quantile level in (0, 1).

    Returns
    -------
    result : RichResult
        Keys: estimate (mean score), scores, n, tau, method.

    References
    ----------
    Gneiting, T. (2011), "Making and evaluating point forecasts",
    Journal of the American Statistical Association 106(494), 746-762;
    arXiv:0912.0902, sec. 3.3, eq. S(x, y) = (1(x >= y) - alpha)
    (g(x) - g(y)) with g the identity, and Theorem 9 [source:
    library/pdf/fetched-wave3/gneiting-2011-quantiles-point-forecasts.pdf].
    Koenker, R. and Bassett, G. (1978), "Regression quantiles",
    Econometrica 46(1), 33-50 (check function).
    """
    tau = float(tau)
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")
    y = np.atleast_1d(np.asarray(y, dtype=float))
    x = np.atleast_1d(np.asarray(y_hat, dtype=float))
    if len(x) == 1 and len(y) > 1:
        x = np.asarray([float(x[0])] * len(y))
    if len(x) != len(y):
        raise ValueError("y and y_hat must have equal length")
    ind = np.asarray([1.0 if float(x[i]) >= float(y[i]) else 0.0 for i in range(len(y))])
    scores = (ind - tau) * (x - y)
    return RichResult(
        payload={
            "estimate": float(np.mean(scores)),
            "scores": scores,
            "n": len(y),
            "tau": tau,
            "method": "Quantile (pinball) loss",
        }
    )


quantile_forecast = qrF


def cheatsheet():
    return "qrF: quantile/pinball loss (Gneiting 2011; Koenker-Bassett 1978)"
