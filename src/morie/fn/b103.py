# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 1.3: mean squared error cost of the linear model."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_mse_cost"]


def burkov_lm_ch1_mse_cost(w, b, x, y, N=None):
    """J(w, b) = mean over the dataset of (wx_i + b - y_i)^2.

    N defaults to len(x); passing a different N is refused rather than
    silently renormalising, since the book's N IS the dataset size.

    References: Burkov LM (2025), Ch 1, Eq 1.3, p. 22 (PDF-verified).

    Examples
    --------
    >>> burkov_lm_ch1_mse_cost(2.0, 0.0, [1.0, 2.0], [2.0, 4.0])["cost"]
    0.0
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape; got {x.shape} and "
            f"{y.shape}.")
    if N is not None and int(N) != len(x):
        raise ValueError(
            f"N = {N} does not match the dataset size {len(x)}; the N in "
            "Eq 1.3 is the dataset size, not a free parameter.")
    resid = float(w) * x + float(b) - y
    cost = float(np.mean(resid ** 2))
    return RichResult(payload={
        "cost": cost, "estimate": cost,
        "residuals": [float(v) for v in resid], "n": len(x),
        "method": "MSE cost J(w, b) (Burkov Eq 1.3)"})


def cheatsheet():
    return "b103: MSE cost of the linear model (Burkov Eq 1.3)"
