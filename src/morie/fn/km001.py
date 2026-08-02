# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.1: the unidirectional encoder recurrence."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_unidirectional_encoder_state"]


def kamath_ch2_unidirectional_encoder_state(h_t_1, x_t, f=None):
    """h_t = f(h_{t-1}, x_t); the default f is the elementwise
    tanh(h + x), the simplest recurrence with the stated signature.
    Pass any callable f(h, x) -> h to use a parameterised cell.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.1, printed
    p. 30 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> kamath_ch2_unidirectional_encoder_state([0.0], [0.0])["h"]
    [0.0]
    """
    h = np.atleast_1d(np.asarray(h_t_1, dtype=float))
    x = np.atleast_1d(np.asarray(x_t, dtype=float))
    if f is None:
        if h.shape != x.shape:
            raise ValueError(
                f"the default cell needs matching shapes; got {h.shape} "
                f"and {x.shape}. Pass a callable f for projected inputs.")
        out = np.tanh(h + x)
    else:
        out = np.atleast_1d(np.asarray(f(h, x), dtype=float))
    return RichResult(payload={
        "h": [float(v) for v in out], "estimate": float(out[0]),
        "n": len(out),
        "method": "Encoder recurrence h_t = f(h_t-1, x_t) (Kamath Eq 2.1)"})


def cheatsheet():
    return "km001: encoder recurrence, default cell tanh(h + x)"
