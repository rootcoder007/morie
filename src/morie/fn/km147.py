# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.19: the output projector."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_output_alignment"]


def kamath_ch9_output_alignment(S_X, out_align=None):
    r"""H_X = OUT_ALIGN_{T->X}(S_X).

    ``out_align`` is the caller's projector: a callable, or a weight
    matrix W applied as ``S_X @ W``. The signal tokens go in, the
    generator-readable features come out; the result must be finite
    and non-empty.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.19, printed
    p. 397.

    Examples
    --------
    >>> out = kamath_ch9_output_alignment([[1.0, 2.0]],
    ...                                   [[0.0], [2.0]])
    >>> out["features"]           # 1*0 + 2*2
    [[4.0]]
    """
    if out_align is None:
        raise ValueError("out_align= is required: the projector "
                         "callable or its weight matrix.")
    S = np.atleast_2d(np.asarray(S_X, dtype=float))
    if callable(out_align):
        H = np.asarray(out_align(S), dtype=float)
    else:
        W = np.atleast_2d(np.asarray(out_align, dtype=float))
        if S.shape[1] != W.shape[0]:
            raise ValueError(
                f"S_X is {S.shape} and W is {W.shape}; the inner "
                "dimensions do not match.")
        H = S @ W
    if H.size == 0 or not np.all(np.isfinite(H)):
        raise ValueError("the output projector returned empty or "
                         "non-finite features.")
    H2 = np.atleast_2d(H)
    return RichResult(payload={
        "estimate": float(np.linalg.norm(H)),
        "features": [[float(v) for v in row] for row in H2],
        "shape": list(H.shape), "n": int(H2.shape[0]),
        "method": "output projector H_X = OUT_ALIGN(S_X) "
                  "(Kamath Eq 9.19)"})


def cheatsheet():
    return "km147: map signal tokens into modality-generator features"
