# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 4.9: the merged KronA weights."""

from . import _array_core as np

from ._richresult import RichResult
from .km059 import _kron

__all__ = ["kamath_ch4_krona_tuned_weights"]


def _tuned(W, A_k, B_k, s):
    """W + s (A_k (x) B_k), shape-checked. km061 imports this."""
    Wm = np.atleast_2d(np.asarray(W, dtype=float))
    K, _, _ = _kron(A_k, B_k)
    if K.shape != Wm.shape:
        raise ValueError(
            f"A_k (x) B_k is {K.shape} but W is {Wm.shape}; the adapter "
            "cannot be merged.")
    s = float(s)
    if not np.isfinite(s):
        raise ValueError("the scaling factor s must be finite.")
    return Wm + s * K, Wm, K, s


def kamath_ch4_krona_tuned_weights(W, A_k, B_k, s):
    """W_tuned = W + s [A_k (x) B_k].

    The merge that makes KronA free at inference: once added, the
    adapter leaves no extra operation behind. s = 0 returns W
    unchanged, which is the identity the tests use.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Eq 4.9, printed
    p. 154.

    Examples
    --------
    >>> out = kamath_ch4_krona_tuned_weights(
    ...     [[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
    ...     [[1.0]], 2.0)
    >>> out["W_tuned"]
    [[3.0, 1.0], [1.0, 3.0]]
    """
    Wt, Wm, K, s = _tuned(W, A_k, B_k, s)
    return RichResult(payload={
        "W_tuned": [[float(v) for v in row] for row in Wt],
        "delta": [[float(v) for v in row] for row in s * K],
        "s": s, "shape": (int(Wt.shape[0]), int(Wt.shape[1])),
        "estimate": float(Wt[0, 0]), "n": int(Wt.size),
        "method": "merged KronA weights (Kamath Eq 4.9)"})


def cheatsheet():
    return "km062: W_tuned = W + s (A_k (x) B_k), merged adapter"
