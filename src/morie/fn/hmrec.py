# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recall (true positive rate, sensitivity) = TP / (TP + FN)."""

from . import _array_core as np

from ._richresult import RichResult
from .recal import recall as _recall

__all__ = ["geron_recall"]


def geron_recall(y_true, y_pred, pos_label=1):
    """
    Recall (true positive rate, sensitivity) = TP / (TP + FN).

    Formula: R = TP / (TP + FN)

    The counting is DELEGATED to the finished implementation
    :func:`morie.fn.recal.recall`; this wrapper adds the confusion counts
    and the precision companion. Recall is the fraction of the actual
    positives that were caught, so it is the metric that matters when a
    miss is expensive (Geron's cancer-screening framing).

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels, same length.
    pos_label : scalar, default 1
        Label treated as positive.

    Returns
    -------
    result : RichResult
        Keys: recall, tp, fn, fp, f1, estimate, n, method.

    Examples
    --------
    >>> r = geron_recall([1, 1, 0, 0], [1, 0, 0, 0])
    >>> r["tp"], r["fn"]
    (1, 1)
    >>> float(r["recall"])
    0.5
    >>> float(geron_recall([1, 1], [1, 1])["recall"])
    1.0

    References
    ----------
    Geron Ch 3
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.size == 0:
        raise ValueError("geron_recall: y_true is empty")
    if yt.size != yp.size:
        raise ValueError(f"geron_recall: y_true has {yt.size} entries but y_pred has {yp.size}")

    res = _recall(yt, yp, pos_label=pos_label)
    tp, fn, fp = int(res.extra["tp"]), int(res.extra["fn"]), int(res.extra["fp"])
    if tp + fn == 0:
        raise ValueError(f"geron_recall: no instance has the positive label {pos_label!r}, so recall is undefined")
    rec = tp / (tp + fn)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp + fp) and (prec + rec) > 0 else 0.0
    return RichResult(
        title="Recall",
        summary_lines=[("Recall", rec), ("TP", tp), ("FN", fn)],
        interpretation="Recall is maximised by predicting everything positive; read it with precision.",
        payload={
            "recall": rec,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "f1": float(f1),
            "estimate": rec,
            "n": int(yt.size),
            "method": "Recall TP/(TP+FN), counts delegated to morie.fn.recal.recall",
        },
    )


def cheatsheet():
    return "hmrec: Recall (true positive rate, sensitivity) = TP / (TP + FN)"
