# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.11: the scaled dot score q.k / sqrt(d_k)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_scaled_dot_score"]


def kamath_ch2_scaled_dot_score(q, k, d_k=None):
    """alpha(q, k) = q.k / sqrt(d_k); d_k defaults to len(q) and a
    supplied d_k that contradicts the vectors is refused -- the scale
    exists to keep the variance of the score at 1, and a wrong d_k
    silently changes the temperature of every downstream softmax.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.11, printed
    p. 33 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> kamath_ch2_scaled_dot_score([2.0, 0.0], [2.0, 0.0])["estimate"]
    2.82842712474619
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if q.shape != k.shape:
        raise ValueError("q and k must have the same dimension.")
    d = len(q) if d_k is None else int(d_k)
    if d != len(q):
        raise ValueError(
            f"d_k = {d} contradicts the vector dimension {len(q)}.")
    return RichResult(payload={
        "estimate": float(np.dot(q, k) / np.sqrt(d)), "d_k": d,
        "n": len(q),
        "method": "Scaled dot score (Kamath Eq 2.11)"})


def cheatsheet():
    return "km011: q.k/sqrt(d_k), contradictory d_k refused"
