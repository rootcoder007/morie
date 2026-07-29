# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ordinal encoding: map categories to integers preserving order."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_ordinal_encoding"]


def geron_ordinal_encoding(X, categories=None):
    """
    Ordinal encoding: map categorical values to integers preserving order.

    Formula: x_cat -> int via lookup table; order matters

    The integer codes assert a ranking AND equal spacing between
    neighbouring levels. That is fine for "bad < average < good" and
    wrong for "Canada, Mexico, Japan", where a model will read the codes
    as distances -- Geron's reason for reaching for one-hot instead. The
    order used is whatever ``categories`` says; sorted order is only a
    default, and an unseen value raises rather than silently mapping to a
    neighbouring code.

    Parameters
    ----------
    X : array-like, shape (m,) or (m, p)
        Categorical values.
    categories : sequence, optional
        Level order (one sequence per column for 2-D input). Defaults to
        the sorted unique values, which is alphabetical, not semantic.

    Returns
    -------
    result : RichResult
        Keys: encoded, categories, n_categories, estimate, n, method.

    Examples
    --------
    >>> r = geron_ordinal_encoding(["low", "high", "mid"],
    ...                            categories=["low", "mid", "high"])
    >>> [int(v) for v in r["encoded"].ravel()]
    [0, 2, 1]

    Without an explicit order the codes are alphabetical, which here
    ranks "high" below "low":

    >>> [int(v) for v in geron_ordinal_encoding(["low", "high"])["encoded"].ravel()]
    [1, 0]

    An unseen level is an error, not a silent code:

    >>> geron_ordinal_encoding(["a", "z"], categories=["a", "b"])
    Traceback (most recent call last):
        ...
    ValueError: geron_ordinal_encoding: column 0 has values not in categories: ['z']

    References
    ----------
    Geron Ch 2
    """
    A = np.asarray(X)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_ordinal_encoding: X must be 1-D or 2-D, got ndim={A.ndim}")
    m, p = A.shape
    if m == 0 or p == 0:
        raise ValueError("geron_ordinal_encoding: X is empty")

    if categories is None:
        cats = [np.unique(A[:, j]) for j in range(p)]
    else:
        cs = list(categories)
        if p == 1 and cs and np.ndim(cs[0]) == 0:
            cs = [cs]
        if len(cs) != p:
            raise ValueError(f"geron_ordinal_encoding: categories has {len(cs)} entries for {p} columns")
        cats = [np.asarray(c) for c in cs]

    enc = np.empty((m, p), dtype=int)
    for j in range(p):
        levels = cats[j]
        if levels.size != np.unique(levels).size:
            raise ValueError(f"geron_ordinal_encoding: column {j} has duplicated categories")
        eq = A[:, j][:, None] == levels[None, :]
        unseen = ~eq.any(axis=1)
        if unseen.any():
            bad = np.unique(A[:, j][unseen])
            raise ValueError(f"geron_ordinal_encoding: column {j} has values not in categories: {bad.tolist()}")
        enc[:, j] = np.argmax(eq, axis=1)

    return RichResult(
        title="Ordinal encoding",
        summary_lines=[("Variables", p), ("Levels per variable", [int(c.size) for c in cats])],
        interpretation="Integer codes imply order AND equal spacing; use one-hot when neither holds.",
        payload={
            "encoded": enc,
            "categories": cats,
            "n_categories": [int(c.size) for c in cats],
            "estimate": enc,
            "n": int(m),
            "method": "Ordinal (integer) encoding against a fixed level order",
        },
    )


def cheatsheet():
    return "hmord: Ordinal encoding of categorical variables"
