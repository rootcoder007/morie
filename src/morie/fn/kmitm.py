# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image-Text Matching (ITM): binary head over a fused image-text
embedding."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_image_text_matching"]


def _sigmoid(z):
    # Branch-stable: exp never sees a large positive argument.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def kamath_image_text_matching(image_emb, text_emb, W, b, fuse=None):
    """p(match | I, T) = sigmoid(w^T fused(I, T) + b).

    The default fusion is concatenation ``[I; T]``, the cheapest thing
    the head can be defined over; pass ``fuse(I, T) -> vector`` for a
    cross-attention fusion. ``W`` must match the fused width, which is
    checked -- a head silently applied to half a vector is the bug
    this guard exists for.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, image-text matching.

    Examples
    --------
    >>> out = kamath_image_text_matching([1.0, 0.0], [0.0, 2.0],
    ...                                  [0.0, 0.0, 0.0, 0.0], 0.0)
    >>> out["estimate"]
    0.5
    >>> out2 = kamath_image_text_matching([1.0], [1.0], [1.0, 1.0], 0.0)
    >>> abs(out2["estimate"] - 1 / (1 + 2.718281828459045 ** -2)) < 1e-12
    True
    """
    I = np.atleast_1d(np.asarray(image_emb, dtype=float)).ravel()
    T = np.atleast_1d(np.asarray(text_emb, dtype=float)).ravel()
    if I.size == 0 or T.size == 0:
        raise ValueError("both embeddings must be non-empty.")
    if fuse is None:
        fused = np.concatenate([I, T])
        how = "concatenation [I; T]"
    else:
        if not callable(fuse):
            raise ValueError("fuse must be callable (I, T) -> vector.")
        fused = np.atleast_1d(np.asarray(fuse(I, T), dtype=float)).ravel()
        how = "caller-supplied fusion"
    w = np.atleast_1d(np.asarray(W, dtype=float)).ravel()
    if w.size != fused.size:
        raise ValueError(
            f"W has {w.size} weights but the fused embedding has "
            f"{fused.size} dimensions ({how}).")
    z = float(np.dot(w, fused) + float(b))
    p = float(_sigmoid(np.array([z]))[0])
    return RichResult(payload={
        "estimate": p, "probability": p, "logit": z,
        "match": bool(p >= 0.5),
        "fused": [float(v) for v in fused], "fusion": how,
        "n": int(fused.size),
        "method": "Image-text matching head p = sigmoid(w.fused + b)"})


def cheatsheet():
    return "kmitm: sigmoid(w^T [I; T] + b) match probability"
