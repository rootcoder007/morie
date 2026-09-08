# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.6: the CLIP text-to-image contrastive loss."""

from ._richresult import RichResult
from .km133 import kamath_ch9_clip_image_to_text

__all__ = ["kamath_ch9_clip_text_to_image"]


def kamath_ch9_clip_text_to_image(L, V, sigma, N=None):
    r"""L_t2i = -(1/N) sum_i log[ exp(L_i.V_i/s) / sum_j exp(L_i.V_j/s) ].

    The softmax is taken over the OTHER modality's row, so Eq 9.6 is
    Eq 9.5 with the arguments exchanged; ``morie.fn.km133`` holds the
    implementation and this passes ``(L, V)`` into it.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.6, printed
    p. 386.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch9_clip_text_to_image([[1.0, 0.0], [0.0, 1.0]],
    ...                                     [[1.0, 0.0], [0.0, 1.0]], 1.0)
    >>> abs(out["estimate"] - (math.log(math.e + 1) - 1)) < 1e-12
    True
    """
    r = kamath_ch9_clip_image_to_text(L, V, sigma, N)
    return RichResult(payload={
        "estimate": r["estimate"], "per_pair": r["per_pair"],
        "logits": r["logits"], "temperature": r["temperature"],
        "n": r["n"],
        "method": "CLIP text-to-image contrastive loss (Kamath Eq 9.6; "
                  "the Eq 9.5 core in km133, modalities swapped)"})


def cheatsheet():
    return "km134: km133 with text as the query modality"
