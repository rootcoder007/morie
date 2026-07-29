# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.7: the total CLIP contrastive loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_clip_contrastive_total"]


def kamath_ch9_clip_contrastive_total(L_i2t, L_t2i):
    r"""L_CL = L_i2t + L_t2i.

    Both halves are cross-entropies and so are non-negative; a
    negative input means one of them was computed with a sign error
    and is rejected here rather than silently summed.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.7, printed
    p. 386.

    Examples
    --------
    >>> out = kamath_ch9_clip_contrastive_total(0.25, 0.75)
    >>> out["estimate"]
    1.0
    """
    a = float(L_i2t)
    b = float(L_t2i)
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("both contrastive losses must be finite.")
    if a < 0 or b < 0:
        raise ValueError("a contrastive cross-entropy cannot be "
                         f"negative; got {a} and {b}.")
    return RichResult(payload={
        "estimate": a + b, "L_i2t": a, "L_t2i": b, "n": 2,
        "method": "total CLIP contrastive loss (Kamath Eq 9.7)"})


def cheatsheet():
    return "km135: L_i2t + L_t2i, both halves checked non-negative"
