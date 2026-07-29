# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bias-variance decomposition of expected prediction error."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_bias_variance_tradeoff"]


def geron_bias_variance_tradeoff(preds, y, f_true=None):
    """
    Bias-variance decomposition of expected prediction error.

    Formula: E[err] = Bias^2 + Variance + Irreducible_noise

    Parameters
    ----------
    preds : array-like, shape (n_models, n_points)
        Predictions of `n_models` models (bootstrap replicates, seeds, folds)
        at the same `n_points` evaluation points.
    y : array-like, shape (n_points,)
        Observed targets at those points.
    f_true : array-like, optional
        Noise-free target function values. When given, bias^2 is measured
        against f_true and the residual E[(y - f_true)^2] is reported as the
        irreducible noise; otherwise noise is whatever the identity leaves
        over (exactly 0 when y is treated as noise-free).

    Returns
    -------
    result : RichResult
        Keys: bias2, variance, noise, mse, mean_pred, estimate, n, method.

    Examples
    --------
    >>> r = geron_bias_variance_tradeoff([[1.0, 2.0], [3.0, 4.0]], [2.0, 2.0])
    >>> float(r["bias2"]), float(r["variance"]), float(r["mse"])
    (0.5, 1.0, 1.5)
    >>> round(float(r["noise"]), 12)
    0.0
    >>> r2 = geron_bias_variance_tradeoff([[2.0, 2.0], [2.0, 2.0]], [2.0, 3.0])
    >>> float(r2["variance"]), float(r2["bias2"])
    (0.0, 0.5)

    References
    ----------
    Géron Ch 1
    """
    P = np.asarray(preds, dtype=float)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    if P.ndim != 2:
        raise ValueError(f"geron_bias_variance_tradeoff: preds must be 2-D (n_models, n_points), got ndim={P.ndim}")
    yv = np.asarray(y, dtype=float).ravel()
    if P.size == 0:
        raise ValueError("geron_bias_variance_tradeoff: preds is empty")
    if P.shape[1] != yv.size:
        raise ValueError(
            f"geron_bias_variance_tradeoff: preds has {P.shape[1]} points but y has {yv.size} entries"
        )
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(yv))):
        raise ValueError("geron_bias_variance_tradeoff: preds and y must be finite")

    mean_pred = P.mean(axis=0)
    # Variance across models is the population variance (ddof=0): the
    # decomposition identity only closes with the biased estimator.
    var_point = P.var(axis=0, ddof=0)
    variance = float(np.mean(var_point))
    mse = float(np.mean((P - yv[None, :]) ** 2))

    target = yv if f_true is None else np.asarray(f_true, dtype=float).ravel()
    if target.size != yv.size:
        raise ValueError(
            f"geron_bias_variance_tradeoff: f_true has {target.size} entries but y has {yv.size}"
        )
    bias2 = float(np.mean((mean_pred - target) ** 2))
    if f_true is None:
        noise = mse - bias2 - variance
    else:
        noise = float(np.mean((yv - target) ** 2))

    return RichResult(
        title="Bias-variance decomposition",
        summary_lines=[("Bias^2", bias2), ("Variance", variance), ("Noise", float(noise)), ("Total MSE", mse)],
        payload={
            "bias2": bias2,
            "variance": variance,
            "noise": float(noise),
            "mse": mse,
            "mean_pred": mean_pred,
            "var_point": var_point,
            "n_models": int(P.shape[0]),
            "estimate": mse,
            "n": int(yv.size),
            "method": "Bias-variance decomposition E[err] = bias^2 + variance + noise",
        },
    )


def cheatsheet():
    return "hmbv: Bias-variance decomposition of expected prediction error"
