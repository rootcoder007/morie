# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 2: bits per character from token-level cross-entropy."""

import math

from ._richresult import RichResult

__all__ = ["burkov_bits_per_character"]


def burkov_bits_per_character(ce_loss, n_tokens, n_characters):
    """BPC = (L_CE * N_tokens) / (ln 2 * N_characters), L_CE in nats.

    References: Burkov LM (2025), Ch 2, bits per character.

    Examples
    --------
    >>> round(burkov_bits_per_character(math.log(2), 100, 100)["estimate"], 12)
    1.0
    """
    l = float(ce_loss); nt = int(n_tokens); nc = int(n_characters)
    if l < 0:
        raise ValueError("cross-entropy cannot be negative.")
    if nt < 1 or nc < 1:
        raise ValueError("token and character counts must be positive.")
    bpc = (l * nt) / (math.log(2.0) * nc)
    return RichResult(payload={
        "estimate": bpc, "bits_per_token": l / math.log(2.0),
        "chars_per_token": nc / nt, "n": nt,
        "method": "Bits per character (Burkov Ch 2)"})


def cheatsheet():
    return "bkbpc: bits per character from cross-entropy (Burkov Ch 2)"
