# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 1: the Euclidean norm."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_vector_norm"]


def burkov_vector_norm(a):
    """||a||_2 = sqrt(sum a_i^2).

    References: Burkov LM (2025), Ch 1, vector norm.

    Examples
    --------
    >>> burkov_vector_norm([3.0, 4.0])["estimate"]
    5.0
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    return RichResult(payload={
        "estimate": float(np.linalg.norm(a)),
        "squared": float(np.dot(a, a)), "n": len(a),
        "method": "L2 norm (Burkov Ch 1)"})


def cheatsheet():
    return "bknrm: Euclidean norm sqrt(sum a_i^2) (Burkov Ch 1)"
