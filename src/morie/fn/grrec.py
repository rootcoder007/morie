# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recall (sensitivity, TPR) = TP / (TP + FN)."""

from . import _array_core as np

from ._richresult import RichResult
from .grcfm import geron_confusion_matrix

__all__ = ["geron_recall"]

_METHOD = "Recall TP/(TP+FN)"


def geron_recall(y_true, y_pred, positive=1, average=None):
    r"""Share of actual positives that were caught.

    .. math::
        \mathrm{recall} = \frac{TP}{TP + FN}

    Same counting engine as precision -- :func:`morie.fn.grcfm.geron_confusion_matrix`
    -- but the denominator is a *row* sum of the confusion matrix rather
    than a column sum, which is the distinction that gets flipped when
    people compute these by hand.  Zero support for the positive class
    raises: recall of a class that never occurs is undefined, not 0.

    Parameters
    ----------
    y_true, y_pred : array-like of int
    positive : int, optional
    average : {None, "macro"}, optional

    Returns
    -------
    RichResult
        Payload keys ``recall``, ``tp``, ``fn``, ``per_class``, ``f1``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 3, Eq 3-2 (Recall).

    Examples
    --------
    Three actual positives, two found:

    >>> yt = [1, 1, 0, 0, 1]
    >>> yp = [1, 0, 1, 0, 1]
    >>> r = geron_recall(yt, yp)
    >>> r["tp"], r["fn"]
    (2, 1)
    >>> round(r["recall"], 6)
    0.666667

    Predicting everything positive gives perfect recall and poor
    precision -- the trade-off in one line:

    >>> geron_recall(yt, [1, 1, 1, 1, 1])["recall"]
    1.0
    """
    yt = np.asarray(y_true).ravel().astype(int)
    yp = np.asarray(y_pred).ravel().astype(int)
    if yt.size == 0:
        raise ValueError("y_true is empty.")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true has {yt.size} labels but y_pred has {yp.size}.")
    n_classes = int(max(yt.max(), yp.max())) + 1
    cm = geron_confusion_matrix(yt, yp, n_classes=n_classes)
    M = np.asarray(cm["matrix"], dtype=float)
    support = M.sum(axis=1)
    per_class = np.asarray(cm["recall"], dtype=float)

    if average is None:
        pos = int(positive)
        if not (0 <= pos < n_classes):
            raise ValueError(f"positive class {pos} not in [0, {n_classes - 1}].")
        if support[pos] == 0:
            raise ValueError(f"class {pos} never occurs in y_true; recall is undefined.")
        tp = int(M[pos, pos])
        fn = int(support[pos] - M[pos, pos])
        val = tp / (tp + fn)
    elif average == "macro":
        seen = support > 0
        val = float(per_class[seen].mean())
        tp = int(np.trace(M))
        fn = int(M.sum() - np.trace(M))
    else:
        raise ValueError(f"average must be None or 'macro', got {average!r}.")

    return RichResult(
        title="Recall",
        summary_lines=[("Recall", float(val)), ("TP", tp), ("FN", fn)],
        payload={
            "recall": float(val),
            "tp": tp,
            "fn": fn,
            "per_class": per_class.tolist(),
            "f1": cm["f1"],
            "estimate": float(val),
            "n": int(yt.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grrec: recall = TP/(TP+FN), a confusion-matrix row sum; counts delegated to grcfm"


# compact alias per ledger/NAMING.md
geronrecall = geron_recall
