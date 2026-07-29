# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.5: WEAT's differential association of one attribute."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_weat_similarity"]


def _rows(X, name):
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.size == 0 or A.shape[0] == 0:
        raise ValueError(f"{name} is empty; a mean over no words is "
                         "undefined.")
    norms = np.linalg.norm(A, axis=1)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero vector; its cosine is "
                         "undefined.")
    return A, norms


def _cos_mean(a, W, name):
    A, n_w = _rows(W, name)
    na = float(np.linalg.norm(a))
    if na == 0:
        raise ValueError("a is a zero vector; its cosine is undefined.")
    if A.shape[1] != a.shape[0]:
        raise ValueError(
            f"{name} has width {A.shape[1]} but a has {a.shape[0]}.")
    return float(np.mean((A @ a) / (n_w * na)))


def _s(a, W_1, W_2):
    """s(a, W_1, W_2) -- imported by km080, km082, km083."""
    a = np.atleast_1d(np.asarray(a, dtype=float))
    return _cos_mean(a, W_1, "W_1") - _cos_mean(a, W_2, "W_2")


def kamath_ch6_weat_similarity(a, W_1, W_2):
    """s(a,W_1,W_2) = mean_{w1} cos(a,w1) - mean_{w2} cos(a,w2).

    How much more one attribute word leans toward the first neutral set
    than the second. Bounded in [-2, 2] since each mean of cosines lies
    in [-1, 1]; the sign, not the scale, carries the association.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.5, printed
    p. 234.

    Examples
    --------
    >>> out = kamath_ch6_weat_similarity([1.0, 0.0], [[1.0, 0.0]],
    ...                                  [[0.0, 1.0]])
    >>> out["estimate"]
    1.0
    >>> kamath_ch6_weat_similarity([1.0, 1.0], [[1.0, 0.0]],
    ...                            [[0.0, 1.0]])["estimate"]
    0.0
    """
    av = np.atleast_1d(np.asarray(a, dtype=float))
    m1 = _cos_mean(av, W_1, "W_1")
    m2 = _cos_mean(av, W_2, "W_2")
    return RichResult(payload={
        "estimate": m1 - m2, "mean_cos_W1": m1, "mean_cos_W2": m2,
        "n": int(av.shape[0]),
        "method": "WEAT differential association s (Kamath Eq 6.5)"})


def cheatsheet():
    return "km081: s(a) = mean cos to W_1 minus mean cos to W_2"
