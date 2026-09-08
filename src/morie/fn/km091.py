# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.15: Demographic Representation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_demographic_representation"]


def _tokens(Y):
    return Y.split() if isinstance(Y, str) else list(Y)


def _count(word, outputs):
    """C(word, Yhat) summed over the outputs; km092 imports this."""
    return sum(_tokens(Y).count(word) for Y in outputs)


def kamath_ch6_demographic_representation(G_i, A_i, Yhat):
    """DR(G_i) = sum_{a in A_i} sum_{Yhat} C(a, Yhat).

    A raw COUNT of how often group G_i's attribute words appear across
    the generated outputs -- the vector over groups is what gets
    normalised and compared to a reference distribution, so this must
    stay a count, not a rate. Per-word counts come back too, since one
    dominant pronoun can carry a whole group's total.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.15, printed
    p. 236.

    Examples
    --------
    >>> out = kamath_ch6_demographic_representation(
    ...     "feminine", ["she", "her"], ["she went", "her cat and she"])
    >>> out["estimate"], out["per_word"]
    (3.0, {'she': 2, 'her': 1})
    """
    words = list(A_i)
    outs = list(Yhat)
    if not words:
        raise ValueError("A_i is empty; a group with no attribute words "
                         "cannot be counted.")
    if not outs:
        raise ValueError("Yhat is empty; there is nothing to count in.")
    per = {a: int(_count(a, outs)) for a in words}
    total = float(sum(per.values()))
    n_tokens = int(sum(len(_tokens(Y)) for Y in outs))
    return RichResult(payload={
        "estimate": total, "group": G_i, "per_word": per,
        "share_of_tokens": (total / n_tokens) if n_tokens else 0.0,
        "n_outputs": len(outs), "n": n_tokens,
        "method": "Demographic Representation count (Kamath Eq 6.15)"})


def cheatsheet():
    return "km091: total count of a group's attribute words in outputs"
