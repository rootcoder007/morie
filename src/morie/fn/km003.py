# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.3: the simplest context mapping, c = h_T."""

import numpy as np

from ._richresult import RichResult
from .km002 import kamath_ch2_context_vector

__all__ = ["kamath_ch2_context_simplest"]


def kamath_ch2_context_simplest(h_T, all_states=None):
    """c = h_T. When the full state stack is supplied the equality
    with Eq 2.2's "last" mapping is verified rather than assumed.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.3, printed
    p. 30 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> kamath_ch2_context_simplest([3.0, 4.0])["context"]
    [3.0, 4.0]
    """
    h = np.atleast_1d(np.asarray(h_T, dtype=float))
    agrees = None
    if all_states is not None:
        via_m = kamath_ch2_context_vector(all_states, "last")["context"]
        agrees = bool(np.allclose(via_m, h))
        if not agrees:
            raise ValueError(
                "h_T does not equal the last row of all_states; the "
                "simplest mapping is c = h_T and these disagree.")
    return RichResult(payload={
        "context": [float(v) for v in h], "agrees_with_eq22": agrees,
        "estimate": float(h[0]), "n": len(h),
        "method": "Simplest context c = h_T (Kamath Eq 2.3)"})


def cheatsheet():
    return "km003: c = h_T, checked against Eq 2.2's last mapping"
