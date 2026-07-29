# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 1.1: the one-feature linear model f(x) = wx + b."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_linear_function"]


def burkov_lm_ch1_linear_function(x, w, b):
    """f(x) = wx + b, elementwise over x.

    References: Burkov LM (2025), Ch 1, Eq 1.1, p. 20 (PDF-verified).

    Examples
    --------
    >>> burkov_lm_ch1_linear_function([1.0, 2.0], 3.0, -1.0)["predictions"]
    [2.0, 5.0]
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    w = float(w); b = float(b)
    y = w * x + b
    return RichResult(payload={
        "predictions": [float(v) for v in y],
        "estimate": float(y[0]), "w": w, "b": b, "n": len(x),
        "method": "Linear model f(x) = wx + b (Burkov Eq 1.1)"})


def cheatsheet():
    return "b101: the one-feature linear model f(x) = wx + b (Burkov Eq 1.1)"
