# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Information gain from a candidate split."""

import numpy as np

from ._richresult import RichResult
from .grent import geron_shannon_entropy
from .grgin import geron_gini_impurity

__all__ = ["geron_information_gain"]

_METHOD = "Information gain of a split"


def geron_information_gain(y, left_mask, criterion="entropy"):
    r"""Impurity drop achieved by one split.

    .. math::
        IG = H(\text{parent})
        - \frac{m_L}{m} H(\text{left})
        - \frac{m_R}{m} H(\text{right})

    The children are weighted by their *size*, not averaged flat.
    Without that weighting CART would happily peel one instance off
    into a perfectly pure leaf and call it a great split.

    Impurity is delegated:
    :func:`morie.fn.grent.geron_shannon_entropy` for
    ``criterion="entropy"``, :func:`morie.fn.grgin.geron_gini_impurity`
    for ``criterion="gini"``.  Information gain is never negative -- a
    weighted average of child impurities cannot exceed the parent's --
    so a negative result would be a bug, and the sign is asserted here.

    Parameters
    ----------
    y : array-like, shape (m,)
        Labels at the parent node.
    left_mask : array-like of bool, shape (m,)
        True for instances sent left. Neither side may be empty.
    criterion : {"entropy", "gini"}, optional

    Returns
    -------
    RichResult
        Payload keys ``information_gain``, ``parent_impurity``,
        ``left_impurity``, ``right_impurity``, ``m_left``, ``m_right``,
        ``weighted_child_impurity``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 5, Information Gain / CART discussion.

    Examples
    --------
    A split that perfectly separates two balanced classes recovers the
    parent's whole bit of entropy:

    >>> r = geron_information_gain([0, 0, 1, 1], [True, True, False, False])
    >>> r["parent_impurity"], r["information_gain"]
    (1.0, 1.0)

    A split that separates nothing gains nothing:

    >>> geron_information_gain([0, 1, 0, 1],
    ...                        [True, True, False, False])["information_gain"]
    0.0

    With Gini the same clean split gains 0.5 -- the scale differs, the
    ranking of splits usually does not:

    >>> geron_information_gain([0, 0, 1, 1], [True, True, False, False],
    ...                        criterion="gini")["information_gain"]
    0.5
    """
    if criterion not in ("entropy", "gini"):
        raise ValueError(f"criterion must be 'entropy' or 'gini', got {criterion!r}.")
    y = np.asarray(y).ravel()
    mask = np.asarray(left_mask)
    if mask.dtype != bool:
        if not np.all(np.isin(mask, (0, 1))):
            raise ValueError("left_mask must be boolean (or 0/1).")
        mask = mask.astype(bool)
    mask = mask.ravel()
    if mask.size != y.size:
        raise ValueError(f"left_mask has {mask.size} entries but y has {y.size}.")
    if y.size == 0:
        raise ValueError("y is empty.")
    mL, mR = int(mask.sum()), int((~mask).sum())
    if mL == 0 or mR == 0:
        raise ValueError(
            f"the split sends every instance to one side (left {mL}, right {mR}); "
            f"information gain is only defined for a split with two non-empty children."
        )

    def imp(arr):
        if criterion == "entropy":
            return geron_shannon_entropy(arr)["entropy"]
        return geron_gini_impurity(arr)["gini"]

    parent = imp(y)
    left = imp(y[mask])
    right = imp(y[~mask])
    m = y.size
    child = (mL / m) * left + (mR / m) * right
    ig = parent - child
    if ig < -1e-12:
        raise ValueError(
            f"information gain came out negative ({ig}); a weighted average of "
            f"child impurities cannot exceed the parent's, so this is a bug."
        )
    ig = float(max(ig, 0.0))

    return RichResult(
        title=f"Information gain ({criterion})",
        summary_lines=[("Gain", ig), ("Parent", parent), ("Children", child)],
        payload={
            "information_gain": ig,
            "parent_impurity": parent,
            "left_impurity": left,
            "right_impurity": right,
            "weighted_child_impurity": child,
            "m_left": mL,
            "m_right": mR,
            "criterion": criterion,
            "estimate": ig,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grig: IG = H(parent) - (mL/m)H(L) - (mR/m)H(R); impurity via grent / grgin"
