# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Linear regression prediction as a weighted sum of features plus bias."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_ch4_linear_regression_prediction"]

_METHOD = "Linear regression prediction (Eq 4-2)"


def geron_ch4_linear_regression_prediction(theta, x):
    r"""Predict :math:`\hat y` from features, Géron Eq 4-2.

    .. math::
        \hat y = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n

    ``theta`` carries the bias in position 0, so ``x`` is *one shorter*
    than ``theta``.  That is the book's convention and it is why this
    differs from :func:`morie.fn.grlinf.geron_linear_layer_forward`,
    where the bias is a separate argument.

    Parameters
    ----------
    theta : array-like, shape (n + 1,)
        ``[theta_0, ..., theta_n]``; ``theta_0`` is the bias.
    x : array-like, shape (n,) or (m, n)
        One instance, or a stack of ``m`` instances.

    Returns
    -------
    RichResult
        Payload keys ``prediction`` (float for one instance, list for a
        stack), ``contributions`` (per-feature ``theta_j x_j`` for a
        single instance), ``bias``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Geron (2026), Ch 4, Eq 4-2, p. 136.

    Examples
    --------
    Bias 1 plus 2*3 plus 3*4:

    >>> r = geron_ch4_linear_regression_prediction([1.0, 2.0, 3.0], [3.0, 4.0])
    >>> r["prediction"]
    19.0
    >>> r["contributions"]
    [6.0, 12.0]

    A batch of two instances:

    >>> geron_ch4_linear_regression_prediction([0.0, 1.0], [[2.0], [5.0]])["prediction"]
    [2.0, 5.0]
    """
    theta = np.asarray(theta, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    if theta.size < 1:
        raise ValueError("theta must contain at least the bias term theta_0.")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(x)):
        raise ValueError("theta and x must be finite.")
    n = theta.size - 1
    batch = x.ndim == 2
    if x.ndim > 2:
        raise ValueError(f"x must be 1-D (one instance) or 2-D (a batch), got ndim {x.ndim}.")
    Xm = x.reshape(1, -1) if not batch else x
    if Xm.shape[1] != n:
        raise ValueError(
            f"x has {Xm.shape[1]} features but theta implies {n} "
            f"(theta[0] is the bias, so len(theta) = n + 1)."
        )

    pred = theta[0] + Xm @ theta[1:]
    contrib = (theta[1:] * Xm[0]).tolist() if not batch else None

    return RichResult(
        title="Linear regression prediction",
        summary_lines=[("Features", int(n)), ("Instances", int(Xm.shape[0]))],
        payload={
            "prediction": pred.tolist() if batch else float(pred[0]),
            "contributions": contrib,
            "bias": float(theta[0]),
            "estimate": pred.tolist() if batch else float(pred[0]),
            "n": int(Xm.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grn002: y_hat = theta_0 + sum_j theta_j x_j -- Geron Eq 4-2"
