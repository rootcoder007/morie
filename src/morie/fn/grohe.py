# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-hot encoding of a categorical feature into indicator columns."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_one_hot_encoding"]

_METHOD = "One-hot encoding"


def geron_one_hot_encoding(categories, levels=None, drop_first=False):
    r"""Turn K categories into K indicator columns.

    .. math::
        e_k(c) = \begin{cases} 1 & c = k\\ 0 & \text{otherwise}\end{cases}

    Unlike ordinal encoding (:mod:`morie.fn.grord`) this imposes no order,
    which is the whole point for nominal features.  The cost is exact
    collinearity -- the columns sum to 1 -- so ``drop_first`` is offered
    for models with an intercept.

    Parameters
    ----------
    categories : sequence
        Category labels, any hashable type.
    levels : sequence, optional
        Category order to use. Values outside it raise.
    drop_first : bool, optional
        Drop the first indicator column (dummy coding).

    Returns
    -------
    RichResult
        Payload keys ``encoded`` (list of rows), ``levels``,
        ``n_columns``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 2, One-hot Encoding section.

    Examples
    --------
    >>> r = geron_one_hot_encoding(["b", "a", "b"])
    >>> r["levels"]
    ['a', 'b']
    >>> r["encoded"]
    [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]

    Rows always sum to 1 (that is the collinearity):

    >>> [sum(row) for row in r["encoded"]]
    [1.0, 1.0, 1.0]
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
            raise ValueError(f"categories contain values absent from levels: {sorted(map(str, missing))}.")
    index = {c: i for i, c in enumerate(lv)}
    M = np.zeros((len(cats), len(lv)))
    M[np.arange(len(cats)), [index[c] for c in cats]] = 1.0
    kept = lv
    if drop_first:
        if len(lv) < 2:
            raise ValueError("drop_first needs at least two levels.")
        M = M[:, 1:]
        kept = lv[1:]

    return RichResult(
        title="One-hot encoding",
        summary_lines=[("Rows", len(cats)), ("Columns", int(M.shape[1]))],
        payload={
            "encoded": M.tolist(),
            "levels": lv,
            "columns": kept,
            "n_columns": int(M.shape[1]),
            "estimate": M.tolist(),
            "n": len(cats),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grohe: one indicator column per level, rows sum to 1; drop_first= for dummy coding"
