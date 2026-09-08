# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.2: the context vector as a mapping of all states."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_context_vector"]


def kamath_ch2_context_vector(h_1_h_T, mapping="mean"):
    """c = m(h_1..h_T); m is "mean", "last", "max" or a callable.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.2, printed
    p. 30 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> kamath_ch2_context_vector([[1.0, 2.0], [3.0, 4.0]])["context"]
    [2.0, 3.0]
    """
    H = np.atleast_2d(np.asarray(h_1_h_T, dtype=float))
    if H.shape[0] == 0:
        raise ValueError("no hidden states supplied.")
    if callable(mapping):
        c = np.atleast_1d(np.asarray(mapping(H), dtype=float))
        name = "callable"
    elif mapping == "mean":
        c = H.mean(axis=0); name = "mean"
    elif mapping == "last":
        c = H[-1]; name = "last"
    elif mapping == "max":
        c = H.max(axis=0); name = "max"
    else:
        raise ValueError(
            f"mapping must be mean, last, max or a callable; got "
            f"{mapping!r}.")
    return RichResult(payload={
        "context": [float(v) for v in c], "mapping": name,
        "estimate": float(c[0]), "n": H.shape[0],
        "method": "Context vector c = m(h_1..h_T) (Kamath Eq 2.2)"})


def cheatsheet():
    return "km002: context vector over encoder states, m selectable"
