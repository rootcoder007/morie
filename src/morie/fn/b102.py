# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 1.2: squared error for a single example."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_squared_error"]


def burkov_lm_ch1_squared_error(y_hat_i, y_i):
    """err(y_hat, y) = (y_hat - y)^2, elementwise over paired inputs.

    References: Burkov LM (2025), Ch 1, Eq 1.2, p. 22 (PDF-verified).

    Examples
    --------
    >>> burkov_lm_ch1_squared_error(3.0, 1.0)["estimate"]
    4.0
    """
    yh = np.atleast_1d(np.asarray(y_hat_i, dtype=float))
    y = np.atleast_1d(np.asarray(y_i, dtype=float))
    if yh.shape != y.shape:
        raise ValueError(
            f"y_hat and y must have the same shape; got {yh.shape} and "
            f"{y.shape}.")
    err = (yh - y) ** 2
    return RichResult(payload={
        "errors": [float(v) for v in err], "estimate": float(err[0]),
        "n": len(y), "method": "Squared error (Burkov Eq 1.2)"})


def cheatsheet():
    return "b102: squared error (y_hat - y)^2 (Burkov Eq 1.2)"
