# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ordinal encoding: map K ordered categories to {0, 1, ..., K-1}."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_ordinal_encoding"]

_METHOD = "Ordinal encoding"


def geron_ordinal_encoding(categories, levels=None):
    r"""Map each category to its rank in the level order.

    .. math::
        \mathrm{enc}(c_k) = k, \qquad k = 0, \dots, K-1

    The encoding asserts that level 2 is "twice" level 1 and that level 3
    sits between 2 and 4.  For a genuinely ordered feature ("bad" <
    "average" < "good") that is information; for a nominal one it is a
    lie, and :mod:`morie.fn.grohe` is the right tool instead.  Because the
    order matters, ``levels`` is honoured verbatim when supplied and only
    falls back to sorting when it is not.

    Parameters
    ----------
    categories : sequence
    levels : sequence, optional
        Explicit low-to-high order.

    Returns
    -------
    RichResult
        Payload keys ``encoded``, ``levels``, ``mapping``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 2, Ordinal Encoding section.

    Examples
    --------
    Declared order is respected even though it is not alphabetical:

    >>> r = geron_ordinal_encoding(["good", "bad", "average"],
    ...                            levels=["bad", "average", "good"])
    >>> r["encoded"]
    [2, 0, 1]
    >>> r["mapping"]["average"]
    1

    Without ``levels`` the order is the sorted order:

    >>> geron_ordinal_encoding(["good", "bad", "average"])["encoded"]
    [2, 1, 0]
    """
    cats = list(categories)
    if not cats:
        raise ValueError("categories is empty; nothing to encode.")
    if levels is None:
        try:
            lv = sorted(set(cats))
        except TypeError as exc:
            raise ValueError(
                "categories are not mutually orderable; pass levels= explicitly."
            ) from exc
    else:
        lv = list(levels)
        if len(set(lv)) != len(lv):
            raise ValueError("levels contains duplicates.")
        missing = set(cats) - set(lv)
        if missing:
            raise ValueError(
                f"categories contain values absent from levels: {sorted(map(str, missing))}."
            )
    mapping = {c: i for i, c in enumerate(lv)}
    enc = [mapping[c] for c in cats]

    return RichResult(
        title="Ordinal encoding",
        summary_lines=[("Rows", len(cats)), ("Levels", len(lv))],
        payload={
            "encoded": enc,
            "levels": lv,
            "mapping": mapping,
            "estimate": enc,
            "n": len(cats),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grord: enc(c_k) = k over the declared level order; use grohe when the feature is nominal"
