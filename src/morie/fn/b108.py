# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 1.8: logistic regression y = sigma(w.x + b)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_logistic_regression"]


def burkov_lm_ch1_logistic_regression(w, x, b):
    """y = 1 / (1 + exp(-(w.x + b))).

    References: Burkov LM (2025), Ch 1, Eq 1.8, p. 40.

    Examples
    --------
    >>> burkov_lm_ch1_logistic_regression([0.0], [1.0], 0.0)["estimate"]
    0.5
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if w.shape != x.shape:
        raise ValueError(
            f"w and x must have the same length; got {len(w)} and "
            f"{len(x)}.")
    z = float(np.dot(w, x) + float(b))
    p = float(1.0 / (1.0 + np.exp(-z)))
    return RichResult(payload={
        "estimate": p, "logit": z, "predicted_class": int(p >= 0.5),
        "n": len(x),
        "method": "Logistic regression sigma(w.x + b) (Burkov Eq 1.8)"})


def cheatsheet():
    return "b108: logistic regression sigma(w.x + b) (Burkov Eq 1.8)"
