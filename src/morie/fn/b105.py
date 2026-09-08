# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 1.5: cosine similarity between two vectors."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_cosine_similarity"]


def burkov_lm_ch1_cosine_similarity(x, y):
    """cos(theta) = x.y / (||x|| ||y||).

    A zero vector has no direction, so similarity with it is refused
    rather than returned as NaN.

    References: Burkov LM (2025), Ch 1, Eq 1.5, p. 31.

    Examples
    --------
    >>> burkov_lm_ch1_cosine_similarity([1.0, 0.0], [0.0, 1.0])["estimate"]
    0.0
    >>> round(burkov_lm_ch1_cosine_similarity([1.0, 2.0],
    ...                                        [2.0, 4.0])["estimate"], 12)
    1.0
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same length; got {len(x)} and "
            f"{len(y)}.")
    nx = float(np.linalg.norm(x)); ny = float(np.linalg.norm(y))
    if nx == 0.0 or ny == 0.0:
        raise ValueError("a zero vector has no direction; cosine "
                         "similarity with it is undefined.")
    c = float(np.dot(x, y) / (nx * ny))
    c = max(-1.0, min(1.0, c))
    return RichResult(payload={
        "estimate": c, "angle_radians": float(np.arccos(c)),
        "n": len(x), "method": "Cosine similarity (Burkov Eq 1.5)"})


def cheatsheet():
    return "b105: cosine similarity x.y/(|x||y|) (Burkov Eq 1.5)"
