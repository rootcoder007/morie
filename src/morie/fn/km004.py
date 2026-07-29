# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.4: the decoder hidden-state recurrence."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_decoder_hidden_state"]


def kamath_ch2_decoder_hidden_state(s_t_1, y_t_1, c, g=None):
    """s_t' = g(s_{t'-1}, y_{t'-1}, c); default g is
    tanh(s + y + c) on matching shapes; pass a callable for a
    parameterised cell.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.4, printed
    p. 31 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> kamath_ch2_decoder_hidden_state([0.0], [0.0], [0.0])["s"]
    [0.0]
    """
    s = np.atleast_1d(np.asarray(s_t_1, dtype=float))
    y = np.atleast_1d(np.asarray(y_t_1, dtype=float))
    c = np.atleast_1d(np.asarray(c, dtype=float))
    if g is None:
        if not (s.shape == y.shape == c.shape):
            raise ValueError(
                "the default cell needs matching shapes; pass a callable "
                "g for projected inputs.")
        out = np.tanh(s + y + c)
    else:
        out = np.atleast_1d(np.asarray(g(s, y, c), dtype=float))
    return RichResult(payload={
        "s": [float(v) for v in out], "estimate": float(out[0]),
        "n": len(out),
        "method": "Decoder recurrence s = g(s, y, c) (Kamath Eq 2.4)"})


def cheatsheet():
    return "km004: decoder recurrence, default cell tanh(s + y + c)"
