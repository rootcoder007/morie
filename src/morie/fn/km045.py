# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 3.4: the Dante cloze probe for parametric knowledge."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch3_dante_cloze"]

MASK = "[MASK]"


def kamath_ch3_dante_cloze(prompt="Dante was born in [MASK]", mask=MASK):
    """Build and CHECK the cloze probe "Dante was born in [MASK]".

    Eq 3.4 is a template, not an arithmetic identity: the only thing
    that can be got wrong is the slot, so that is what is validated.
    Exactly one mask must be present -- zero masks is not a cloze
    prompt and two masks makes the fill ambiguous.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, Eq 3.4, printed
    p. 95.

    Examples
    --------
    >>> out = kamath_ch3_dante_cloze()
    >>> out["mask_index"], out["n"]
    (4, 5)
    >>> kamath_ch3_dante_cloze("Warsaw is the capital of [MASK].")["prompt"]
    'Warsaw is the capital of [MASK].'
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    if prompt.count(mask) != 1:
        raise ValueError(
            f"a cloze prompt needs exactly one {mask}; this one has "
            f"{prompt.count(mask)}.")
    tokens = prompt.split()
    idx = [i for i, t in enumerate(tokens) if mask in t]
    return RichResult(payload={
        "prompt": prompt, "mask": mask, "mask_index": int(idx[0]),
        "tokens": tokens, "estimate": float(idx[0]), "n": len(tokens),
        "method": "cloze knowledge probe (Kamath Eq 3.4)"})


def cheatsheet():
    return "km045: 'Dante was born in [MASK]' -- one slot, validated"
