# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 2: n-gram backoff."""

from ._richresult import RichResult

__all__ = ["burkov_ngram_backoff"]


def burkov_ngram_backoff(counts_by_order, alpha=0.4):
    """Use the highest order whose count is positive, discounting by
    alpha for every level backed off.

    ``counts_by_order`` is highest order first, each entry a
    ``(count_ngram, count_prefix)`` pair. Returns the probability and
    the order actually used.

    References: Burkov LM (2025), Ch 2, backoff (stupid-backoff form).

    Examples
    --------
    >>> burkov_ngram_backoff([(0, 5), (2, 8)], alpha=0.4)["estimate"]
    0.1
    """
    a = float(alpha)
    if not 0 < a <= 1:
        raise ValueError(f"alpha must lie in (0, 1]; got {alpha}.")
    if not counts_by_order:
        raise ValueError("no orders supplied.")
    discount = 1.0
    for level, pair in enumerate(counts_by_order):
        c, p = float(pair[0]), float(pair[1])
        if c < 0 or p < 0:
            raise ValueError("counts must be non-negative.")
        if c > p:
            raise ValueError("count(ngram) cannot exceed count(prefix).")
        if c > 0:
            return RichResult(payload={
                "estimate": discount * c / p, "order_used": level,
                "backed_off": level, "discount": discount, "n": int(p),
                "method": "N-gram backoff (Burkov Ch 2)"})
        discount *= a
    raise ValueError(
        "every order has count 0, including the lowest; backoff has "
        "nowhere left to go. Supply a unigram floor with a positive "
        "count.")


def cheatsheet():
    return "bkbkof: n-gram backoff with per-level discount (Burkov Ch 2)"
