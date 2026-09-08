# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 1: the unit vector."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_unit_vector"]


def burkov_unit_vector(a):
    """a_hat = a / ||a||_2; the zero vector is refused.

    References: Burkov LM (2025), Ch 1, unit vector.

    Examples
    --------
    >>> burkov_unit_vector([3.0, 4.0])["unit"]
    [0.6, 0.8]
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = float(np.linalg.norm(a))
    if n == 0.0:
        raise ValueError("the zero vector has no direction and cannot "
                         "be normalised.")
    u = a / n
    return RichResult(payload={
        "unit": [float(v) for v in u], "estimate": float(u[0]),
        "norm": n, "n": len(a),
        "method": "Unit vector a/||a|| (Burkov Ch 1)"})


def cheatsheet():
    return "bkunit: unit vector a/||a||_2 (Burkov Ch 1)"
