# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.16: the multi-head combination."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_multihead_concat"]


def kamath_ch2_multihead_concat(heads, W_O):
    """multihead = concat(head_1..head_h) W_O (row convention of the
    book's W^O concat). Heads must share their row count.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.16, printed
    p. 36 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> out = kamath_ch2_multihead_concat(
    ...     [[[1.0]], [[2.0]]], [[1.0], [1.0]])
    >>> out["output"]
    [[3.0]]
    """
    hs = [np.atleast_2d(np.asarray(h, dtype=float)) for h in heads]
    if not hs:
        raise ValueError("no heads supplied.")
    rows = hs[0].shape[0]
    if any(h.shape[0] != rows for h in hs):
        raise ValueError("every head must have the same number of rows.")
    concat = np.concatenate(hs, axis=1)
    Wo = np.atleast_2d(np.asarray(W_O, dtype=float))
    if concat.shape[1] != Wo.shape[0]:
        raise ValueError(
            f"concatenated width {concat.shape[1]} does not match W_O's "
            f"{Wo.shape[0]} rows.")
    out = concat @ Wo
    return RichResult(payload={
        "output": [[float(v) for v in r] for r in out],
        "heads": len(hs), "estimate": float(out[0, 0]), "n": rows,
        "method": "Multi-head concat + output projection "
                  "(Kamath Eq 2.16)"})


def cheatsheet():
    return "km016: concat heads then project with W_O"
