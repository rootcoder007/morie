# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.2: AlignScore's alignment function f: (a, b) -> y."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_alignment_function"]

SPACES = {
    "bin": ("ALIGNED", "NOT ALIGNED"),
    "3way": ("ALIGNED", "CONTRADICT", "NEUTRAL"),
}


def kamath_ch6_alignment_function(a, b, y, f=None):
    """f: (a, b) -> y, checked against the declared output space.

    Eq 6.2 is a TYPE, not an arithmetic identity, so what can be got
    wrong is the label space. ``y`` names the head -- "bin"
    {ALIGNED, NOT ALIGNED}, "3way" {ALIGNED, CONTRADICT, NEUTRAL}, or
    "reg" for a value in [0, 1] -- and ``f`` is the caller's trained
    alignment function, applied to (a, b) and validated. There is no
    default f: a definition cannot classify text.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.2, printed
    p. 222.

    Examples
    --------
    >>> out = kamath_ch6_alignment_function("the cat sat", "a cat sat",
    ...     "3way", f=lambda a, b: "ALIGNED")
    >>> out["label"], out["estimate"]
    ('ALIGNED', 1.0)
    >>> kamath_ch6_alignment_function("x", "y", "reg",
    ...                               f=lambda a, b: 0.25)["estimate"]
    0.25
    """
    if f is None:
        raise ValueError(
            "Eq 6.2 defines the SHAPE of the alignment function; supply a "
            "trained f as f=(a, b) -> label.")
    if not callable(f):
        raise ValueError("f must be callable.")
    if y not in SPACES and y != "reg":
        raise ValueError(
            f"unknown output space {y!r}; use 'bin', '3way' or 'reg'.")
    out = f(a, b)
    if y == "reg":
        v = float(out)
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"the regression head returned {v:.6g}; y_reg lies in "
                "[0, 1].")
        label, est = None, v
    else:
        if out not in SPACES[y]:
            raise ValueError(
                f"{out!r} is not in the {y} label space {SPACES[y]!r}.")
        label, est = out, 1.0 if out == "ALIGNED" else 0.0
    return RichResult(payload={
        "estimate": est, "label": label, "space": y,
        "labels": list(SPACES[y]) if y in SPACES else None,
        "a": a, "b": b, "n": 2,
        "method": "AlignScore alignment function (Kamath Eq 6.2)"})


def cheatsheet():
    return "km078: f:(a,b)->y with the label space enforced"
