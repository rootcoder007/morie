# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.13/2.14: sinusoidal positional encodings."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_positional_encoding_sin"]


def kamath_ch2_positional_encoding_sin(i, j, d):
    """P[i, 2j] = sin(i / 10000^(2j/d)).

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.13, printed
    p. 35 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> kamath_ch2_positional_encoding_sin(0, 0, 4)["estimate"]
    0.0
    """
    i = int(i); j = int(j); d = int(d)
    if d < 1:
        raise ValueError("the model dimension d must be positive.")
    if i < 0 or j < 0:
        raise ValueError("position and index must be non-negative.")
    if 2 * j >= d:
        raise ValueError(
            f"2j = {2 * j} must lie below d = {d}; the pair (sin, cos) "
            "fills dimensions 2j and 2j+1.")
    val = float(np.sin(i / 10000.0 ** (2.0 * j / d)))
    return RichResult(payload={
        "estimate": val, "wavelength": float(2 * np.pi
                                             * 10000.0 ** (2.0 * j / d)),
        "n": d,
        "method": "Sinusoidal positional encoding, even dims "
                  "(Kamath Eq 2.13)"})


def cheatsheet():
    return "km013: P[i,2j] = sin(i/10000^(2j/d))"
