# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.14: the cosine half of the positional encoding."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_positional_encoding_cos"]


def kamath_ch2_positional_encoding_cos(i, j, d):
    """P[i, 2j+1] = cos(i / 10000^(2j/d)); shares the wavelength of
    Eq 2.13's sine, so sin^2 + cos^2 = 1 per (i, j) -- the tests
    assert that identity across a grid.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.14, printed
    p. 35 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> kamath_ch2_positional_encoding_cos(0, 0, 4)["estimate"]
    1.0
    """
    i = int(i); j = int(j); d = int(d)
    if d < 1:
        raise ValueError("the model dimension d must be positive.")
    if i < 0 or j < 0:
        raise ValueError("position and index must be non-negative.")
    if 2 * j + 1 >= d:
        raise ValueError(
            f"2j+1 = {2 * j + 1} must lie below d = {d}.")
    val = float(np.cos(i / 10000.0 ** (2.0 * j / d)))
    return RichResult(payload={
        "estimate": val, "n": d,
        "method": "Sinusoidal positional encoding, odd dims "
                  "(Kamath Eq 2.14)"})


def cheatsheet():
    return "km014: P[i,2j+1] = cos(i/10000^(2j/d))"
