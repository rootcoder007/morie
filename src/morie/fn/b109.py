# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 1.9: binary cross-entropy for one example."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_binary_cross_entropy"]


def burkov_lm_ch1_binary_cross_entropy(y_hat_i, y_i):
    """loss = -[y log(y_hat) + (1 - y) log(1 - y_hat)].

    A prediction of exactly 0 or 1 gives infinite loss when it is
    wrong; that is the mathematics, and it is returned as inf rather
    than clipped away silently.

    References: Burkov LM (2025), Ch 1, Eq 1.9, p. 40.

    Examples
    --------
    >>> round(burkov_lm_ch1_binary_cross_entropy(0.5, 1.0)["estimate"], 10)
    0.6931471806
    """
    yh = np.atleast_1d(np.asarray(y_hat_i, dtype=float))
    y = np.atleast_1d(np.asarray(y_i, dtype=float))
    if yh.shape != y.shape:
        raise ValueError(
            f"y_hat and y must have the same shape; got {yh.shape} and "
            f"{y.shape}.")
    if np.any((yh < 0) | (yh > 1)):
        raise ValueError("predicted probabilities must lie in [0, 1].")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("targets must be 0 or 1 for Eq 1.9.")
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = -(y * np.log(yh) + (1.0 - y) * np.log(1.0 - yh))
    loss = np.where(np.isnan(loss), 0.0, loss)   # 0*log0 limit is 0
    return RichResult(payload={
        "losses": [float(v) for v in loss], "estimate": float(loss[0]),
        "mean_loss": float(np.mean(loss)), "n": len(y),
        "method": "Binary cross-entropy (Burkov Eq 1.9)"})


def cheatsheet():
    return "b109: binary cross-entropy loss (Burkov Eq 1.9)"
