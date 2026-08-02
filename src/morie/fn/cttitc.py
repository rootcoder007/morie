# morie.fn -- function file (rootcoder007/morie)
"""Corrected item-total correlation (Nunnally & Bernstein 1994)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ctt_item_total_corr"]


def ctt_item_total_corr(X, item_index):
    r"""Correlation between one item and the total of the *remaining* items.

    .. math::

        r_{i(T-i)} = \operatorname{corr}\!\left(X_i,\ \sum_{j \ne i} X_j\right)

    Parameters
    ----------
    X : array-like, shape (n_respondents, n_items)
        Item response matrix.
    item_index : int
        Zero-based column index of the item under scrutiny.

    Returns
    -------
    RichResult
        keys: ``estimate`` (the corrected correlation), ``uncorrected``
        (against the full total, for comparison), ``item_index``,
        ``n``, ``n_items``, ``method``.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, if there are fewer than 2 items, if
        ``item_index`` is out of range, or if the item or the rest-total has
        zero variance.

    References
    ----------
    Nunnally, J. C., & Bernstein, I. H. (1994). *Psychometric Theory*,
        3rd ed. McGraw-Hill. Item analysis; the corrected (part-whole
        adjusted) item-total correlation.

    Notes
    -----
    "Corrected" means the item is **excluded from the total** it is
    correlated against. The uncorrected version correlates :math:`X_i` with a
    sum that contains :math:`X_i`, so the item is partly correlated with
    itself; that inflation is severe for short scales -- with :math:`k` items
    of equal variance and zero true intercorrelation it is about
    :math:`1/\sqrt{k}`, i.e. **0.45 for a 5-item scale** where the honest
    answer is 0. Both values are returned so the size of the correction is
    visible rather than assumed.
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D (respondents x items); got shape {arr.shape}")
    n, k = arr.shape
    if k < 2:
        raise ValueError(
            f"need at least 2 items to form a rest-total; got {k}. With one item "
            "the 'total minus the item' is empty."
        )
    if n < 2:
        raise ValueError(f"need at least 2 respondents; got {n}")
    item_index = int(item_index)
    if not (0 <= item_index < k):
        raise ValueError(f"item_index {item_index} out of range for {k} items")
    item = arr[:, item_index]
    rest = arr.sum(axis=1) - item
    if np.std(item) == 0.0:
        raise ValueError(
            f"item {item_index} has zero variance -- every respondent gave the same "
            "answer, so no correlation is defined."
        )
    if np.std(rest) == 0.0:
        raise ValueError(
            "the rest-total has zero variance, so no correlation is defined."
        )
    corrected = float(np.corrcoef(item, rest)[0, 1])
    total = arr.sum(axis=1)
    uncorrected = (
        float(np.corrcoef(item, total)[0, 1]) if np.std(total) > 0 else float("nan")
    )
    return RichResult(
        payload={
            "estimate": corrected,
            "uncorrected": uncorrected,
            "item_index": item_index,
            "n": int(n),
            "n_items": int(k),
            "method": "corrected item-total correlation (Nunnally & Bernstein 1994)",
        }
    )


def cheatsheet():
    return "cttitc: corr(item_i, total - item_i), part-whole corrected (Nunnally & Bernstein 1994)."
