# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 2.1: categorical cross-entropy with a one-hot target."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch2_categorical_cross_entropy"]


def burkov_lm_ch2_categorical_cross_entropy(y_hat, c):
    """loss = -log(y_hat[c]) for the correct class c (0-based).

    y_hat must be a probability distribution; a vector that does not
    sum to 1 is refused, because -log of an unnormalised score is not
    the cross-entropy however much it looks like it.

    References: Burkov LM (2025), Ch 2, Eq 2.1, p. 57.

    Examples
    --------
    >>> round(burkov_lm_ch2_categorical_cross_entropy(
    ...     [0.7, 0.2, 0.1], 0)["estimate"], 10)
    0.3566749439
    """
    p = np.atleast_1d(np.asarray(y_hat, dtype=float))
    c = int(c)
    if not 0 <= c < len(p):
        raise ValueError(f"class {c} is out of range for {len(p)} classes.")
    if np.any(p < 0) or abs(float(p.sum()) - 1.0) > 1e-8:
        raise ValueError(
            "y_hat must be a probability distribution (non-negative, "
            f"summing to 1); it sums to {float(p.sum()):.6g}.")
    loss = float(-np.log(p[c])) if p[c] > 0 else float("inf")
    return RichResult(payload={
        "estimate": loss, "p_correct": float(p[c]), "n_classes": len(p),
        "n": len(p),
        "method": "Categorical cross-entropy -log p_c (Burkov Eq 2.1)"})


def cheatsheet():
    return "b201: categorical cross-entropy -log(y_hat[c]) (Burkov Eq 2.1)"
