# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.3: the input projector."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_input_projector"]


def kamath_ch9_input_projector(F_X, in_align=None):
    r"""P_X = IN_ALIGN_{X->T}(F_X).

    ``in_align`` is the caller's projector: either a callable, or a
    weight matrix W applied as ``F_X @ W`` (the linear projector most
    MMLLMs use). The result must be finite and non-empty.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.3, printed
    p. 380.

    Examples
    --------
    >>> out = kamath_ch9_input_projector([[1.0, 2.0]],
    ...                                  [[1.0], [3.0]])
    >>> out["prompts"]              # 1*1 + 2*3
    [[7.0]]
    """
    if in_align is None:
        raise ValueError("in_align= is required: the projector "
                         "callable or its weight matrix.")
    F = np.atleast_2d(np.asarray(F_X, dtype=float))
    if callable(in_align):
        P = np.asarray(in_align(F), dtype=float)
    else:
        W = np.atleast_2d(np.asarray(in_align, dtype=float))
        if F.shape[1] != W.shape[0]:
            raise ValueError(
                f"F_X is {F.shape} and W is {W.shape}; the inner "
                "dimensions do not match.")
        P = F @ W
    if P.size == 0 or not np.all(np.isfinite(P)):
        raise ValueError("the projector returned empty or non-finite "
                         "prompt features.")
    P2 = np.atleast_2d(P)
    return RichResult(payload={
        "estimate": float(np.linalg.norm(P)),
        "prompts": [[float(v) for v in row] for row in P2],
        "shape": list(P.shape), "n": int(P2.shape[0]),
        "method": "input projector P_X = IN_ALIGN(F_X) "
                  "(Kamath Eq 9.3)"})


def cheatsheet():
    return "km131: project modality features into the LLM prompt space"
