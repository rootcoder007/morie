# morie.fn -- slice k04 (rootcoder007/morie)
"""Best-linear-predictor calibration test for heterogeneous treatment effects.

Source FETCHED (reference implementation): ``test_calibration`` in the
CRAN package ``grf`` (Tibshirani, Athey, Wager et al., grf 2.6.1, file
``R/forest_summary.R``).  It implements the calibration check of
Chernozhukov, Demirer, Duflo and Fernandez-Val (2018), "Generic machine
learning inference on heterogeneous treatment effects", arXiv
1712.04802, in the form Athey and Wager (2019) use.  The package source
regresses the residualised outcome on two constructed regressors, with
no intercept::

    target                         = Y - Yhat
    mean.forest.prediction         = (W - What) * mean(tauhat)
    differential.forest.prediction = (W - What) * (tauhat - mean(tauhat))

    lm(target ~ mean + differential + 0),  HC3 sandwich SEs,
    p-values converted to one-sided.

The coefficient on the mean term tests whether the average treatment
effect is correctly estimated; the coefficient on the differential term
is the one this function is named for -- if the CATE estimates carry
real signal that coefficient is 1, and a coefficient far from 1, or a
one-sided p-value that fails to reject 0, says the heterogeneity
estimates are biased or noise.

The HC3 variance is written out rather than delegated, since neither
``sandwich`` nor an equivalent is a dependency here::

    V = (X'X)^-1 [ sum_i x_i x_i' e_i^2 / (1 - h_ii)^2 ] (X'X)^-1

with h_ii the leverage.  That is MacKinnon and White (1985) HC3, the
default ``vcov.type`` in ``test_calibration``.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

import math

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["hettest_bias"]


def _hc3(X, resid):
    """MacKinnon-White (1985) HC3 sandwich covariance."""
    XtX_inv = np.linalg.inv(X.T @ X)
    H = X @ XtX_inv @ X.T
    n = X.shape[0]
    meat = np.zeros((X.shape[1], X.shape[1]))
    for i in range(n):
        h = float(H[i, i])
        w = float(resid[i]) ** 2 / (1.0 - h) ** 2
        xi = X[i, :]
        meat = meat + w * np.outer(xi, xi)
    return XtX_inv @ meat @ XtX_inv


def hettest_bias(y, D, tau_hat, y_hat=None, w_hat=None):
    """Calibration test for CATE estimates.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Treatment assignment.
    tau_hat : array-like, shape (n,)
        Out-of-fold CATE predictions, one per unit.  The third argument
        of the pasted stub was called ``X``; it is the CATE prediction
        vector that the best linear predictor is calibrated against, not
        a covariate matrix.
    y_hat : array-like, optional
        Out-of-fold predictions of E[Y | X].  Defaults to the sample
        mean of ``y``, which is what residualising with no covariate
        model amounts to.
    w_hat : array-like, optional
        Out-of-fold propensity scores E[W | X].  Defaults to the sample
        mean of ``D``.

    Returns
    -------
    RichResult
        keys: ``coef_mean``, ``coef_differential``, ``se_mean``,
        ``se_differential``, ``t_mean``, ``t_differential``,
        ``p_mean``, ``p_differential`` (one-sided, as in grf), ``n``,
        ``method``.
    """
    y = np.asarray(y, dtype=float).ravel()
    d = np.asarray(D, dtype=float).ravel()
    tau = np.asarray(tau_hat, dtype=float).ravel()
    n = int(y.size)
    if d.size != n or tau.size != n:
        raise ValueError("y, D and tau_hat must have the same length")
    if n < 4:
        raise ValueError("need n >= 4")

    yh = np.full(n, float(np.mean(y))) if y_hat is None else np.asarray(y_hat, dtype=float).ravel()
    wh = np.full(n, float(np.mean(d))) if w_hat is None else np.asarray(w_hat, dtype=float).ravel()

    mean_tau = float(np.mean(tau))
    target = y - yh
    wres = d - wh
    r_mean = wres * mean_tau
    r_diff = wres * (tau - mean_tau)
    X = np.column_stack([r_mean, r_diff])
    if float(np.sum(r_diff * r_diff)) <= 0.0:
        raise ValueError("tau_hat is constant: the differential regressor is identically zero")

    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    V = _hc3(X, resid)
    se = np.array([math.sqrt(abs(float(V[0, 0]))), math.sqrt(abs(float(V[1, 1])))])
    tstat = np.array([float(beta[j]) / float(se[j]) if se[j] > 0.0 else float("nan") for j in range(2)])
    df = n - 2

    def one_sided(t):
        if t != t:
            return float("nan")
        two = 2.0 * float(stats.t.sf(abs(t), df))
        # grf: p <- if (t < 0) 1 - p/2 else p/2
        return 1.0 - two / 2.0 if t < 0 else two / 2.0

    return RichResult(
        payload={
            "coef_mean": float(beta[0]),
            "coef_differential": float(beta[1]),
            "se_mean": float(se[0]),
            "se_differential": float(se[1]),
            "t_mean": float(tstat[0]),
            "t_differential": float(tstat[1]),
            "p_mean": one_sided(float(tstat[0])),
            "p_differential": one_sided(float(tstat[1])),
            "n": n,
            "method": "BLP calibration test, HC3 (grf::test_calibration; Chernozhukov et al. 2018)",
        }
    )


def cheatsheet():
    return "htbias: BLP calibration test for CATE estimates"


# compact alias per ledger/NAMING.md
hettestbias = hettest_bias
