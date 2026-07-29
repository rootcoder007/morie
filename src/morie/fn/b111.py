# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 1.11: closed-form BCE gradients for logistic regression."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_bce_gradients"]


def burkov_lm_ch1_bce_gradients(y_hat, y, x, N=None, j=None):
    """d/dw_j = mean((y_hat - y) x_j); d/db = mean(y_hat - y).

    Returns the full weight gradient; `j` selects one coordinate for
    the payload's `estimate` (0-based). N must equal the dataset size.

    References: Burkov LM (2025), Ch 1, Eq 1.11, p. 42.

    Examples
    --------
    >>> out = burkov_lm_ch1_bce_gradients([0.5], [1.0], [[2.0]])
    >>> out["grad_w"]
    [-1.0]
    >>> out["grad_b"]
    -0.5
    """
    yh = np.atleast_1d(np.asarray(y_hat, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != len(yh):
        X = X.T
    if X.shape[0] != len(yh) or len(y) != len(yh):
        raise ValueError(
            f"need one row of x per example; got x {X.shape}, "
            f"y_hat {len(yh)}, y {len(y)}.")
    if N is not None and int(N) != len(y):
        raise ValueError(
            f"N = {N} does not match the dataset size {len(y)}.")
    resid = yh - y
    gw = (X * resid[:, None]).mean(axis=0)
    gb = float(resid.mean())
    est = float(gw[int(j)]) if j is not None else float(gw[0])
    return RichResult(payload={
        "grad_w": [float(v) for v in gw], "grad_b": gb, "estimate": est,
        "n": len(y),
        "method": "BCE gradients for logistic regression (Burkov Eq 1.11)"})


def cheatsheet():
    return "b111: closed-form BCE gradients (Burkov Eq 1.11)"
