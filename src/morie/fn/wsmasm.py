# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MLE asymptotic normality."""

from . import _array_core as np

from ._richresult import RichResult
from .wsmfis import wasserman_fisher_info

__all__ = ["wasserman_mle_asymptotic"]


def wasserman_mle_asymptotic(data, f, theta_hat, x_grid=None):
    """
    Asymptotic distribution of the MLE.

    Formula: sqrt(n)(theta_hat - theta) ~> N(0, 1/I(theta)), so
    se(theta_hat) ~= 1/sqrt(n I(theta_hat)). The information is
    computed numerically at theta_hat via wsmfis (plug-in), and a
    95 percent Wald interval theta_hat +/- 1.959963984540054 se is
    reported.

    Parameters
    ----------
    data : array-like
        Sample (n >= 1); only its size enters the se.
    f : callable or None
        Density f(x, theta); None = exponential model.
    theta_hat : float
        The MLE.
    x_grid : array-like, optional
        Support grid for a custom f (see wsmfis).

    Returns
    -------
    result : dict
        Keys: estimate (theta_hat), se, information, ci_lower,
        ci_upper, n, method.

    References
    ----------
    Wasserman (2004), Ch 9, Theorem 9.18.

    Examples
    --------
    Exponential, theta_hat = 2, n = 100: se ~= theta/sqrt(n) = 0.2.

    >>> out = wasserman_mle_asymptotic(list(range(100)), None, 2.0)
    >>> abs(out["se"] - 0.2) < 1e-4
    True
    >>> out["ci_lower"] < 2.0 < out["ci_upper"]
    True
    >>> round(out["ci_upper"] - out["ci_lower"], 3)
    0.784
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = data.size
    if n == 0:
        raise ValueError("asymptotics for an empty sample are undefined.")
    theta_hat = float(theta_hat)
    info = wasserman_fisher_info(f, theta_hat, x_grid=x_grid)["estimate"]
    se = 1.0 / float(np.sqrt(n * info))
    z = 1.959963984540054
    return RichResult(payload={
        "estimate": theta_hat, "se": se, "information": float(info),
        "ci_lower": theta_hat - z * se, "ci_upper": theta_hat + z * se,
        "n": int(n),
        "method": "MLE se = 1/sqrt(n I(theta_hat)), Wald 95 CI"})


def cheatsheet():
    return "wsmasm: se = 1/sqrt(n I); I numeric at theta_hat via wsmfis"
