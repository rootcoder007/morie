# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.16: Stereotypical Associations."""

import numpy as np

from ._richresult import RichResult
from .km091 import _count, _tokens

__all__ = ["kamath_ch6_stereotypical_assoc"]


def kamath_ch6_stereotypical_assoc(w, A_i, Yhat):
    """ST(w)_i = sum_{a in A_i} sum_{Yhat} C(a,Yhat) I(C(w,Yhat) > 0).

    Demographic Representation (km091) restricted to the outputs that
    actually MENTION the stereotyped word w -- the indicator gates
    whole outputs, not individual attribute tokens, which is why an
    output containing w five times counts no more than one containing
    it once. The counting is km091's, delegated.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.16, printed
    p. 237.

    Examples
    --------
    >>> out = kamath_ch6_stereotypical_assoc(
    ...     "nurse", ["she"], ["she is a nurse", "she is a doctor"])
    >>> out["estimate"], out["n_outputs_with_w"]
    (1.0, 1)
    >>> kamath_ch6_stereotypical_assoc("nurse", ["she"],
    ...     ["she and she are nurse", "she is a doctor"])["estimate"]
    2.0
    """
    words = list(A_i)
    outs = list(Yhat)
    if not words:
        raise ValueError("A_i is empty; a group with no attribute words "
                         "cannot be counted.")
    if not outs:
        raise ValueError("Yhat is empty; there is nothing to count in.")
    gated = [Y for Y in outs if _tokens(Y).count(w) > 0]
    per = {a: int(_count(a, gated)) for a in words}
    total = float(sum(per.values()))
    return RichResult(payload={
        "estimate": total, "word": w, "per_attribute": per,
        "n_outputs_with_w": len(gated), "n": len(outs),
        "method": "Stereotypical Associations count (Kamath Eq 6.16)"})


def cheatsheet():
    return "km092: attribute counts, but only in outputs mentioning w"
