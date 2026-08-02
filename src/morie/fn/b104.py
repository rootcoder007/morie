# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 1.4: the linear model in vector form, w . x + b."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_linear_vector"]


def burkov_lm_ch1_linear_vector(w, x, b):
    """y = w . x + b for one feature vector.

    References: Burkov LM (2025), Ch 1, Eq 1.4, p. 29.

    Examples
    --------
    >>> burkov_lm_ch1_linear_vector([1.0, 2.0], [3.0, 4.0], 0.5)["estimate"]
    11.5
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if w.shape != x.shape:
        raise ValueError(
            f"w and x must have the same length; got {len(w)} and "
            f"{len(x)}.")
    y = float(np.dot(w, x) + float(b))
    return RichResult(payload={
        "estimate": y, "dot": float(np.dot(w, x)), "b": float(b),
        "n": len(x), "method": "Linear model y = w.x + b (Burkov Eq 1.4)"})


def cheatsheet():
    return "b104: vector-form linear model w.x + b (Burkov Eq 1.4)"
