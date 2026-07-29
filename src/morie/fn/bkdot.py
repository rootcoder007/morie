# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 1: the dot product."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_dot_product"]


def burkov_dot_product(a, b):
    """a . b = sum a_i b_i.

    References: Burkov LM (2025), Ch 1, dot product.

    Examples
    --------
    >>> burkov_dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])["estimate"]
    32.0
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if a.shape != b.shape:
        raise ValueError(
            f"vectors must have the same length; got {len(a)} and "
            f"{len(b)}.")
    return RichResult(payload={
        "estimate": float(np.dot(a, b)), "n": len(a),
        "method": "Dot product (Burkov Ch 1)"})


def cheatsheet():
    return "bkdot: dot product sum a_i b_i (Burkov Ch 1)"
