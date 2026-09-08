# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Precision = TP / (TP + FP)."""

from . import _array_core as np

from ._richresult import RichResult
from .precn import precision as _precision

__all__ = ["geron_precision"]


def geron_precision(y_true, y_pred, pos_label=1):
    """
    Precision = TP / (TP + FP).

    Formula: P = TP / (TP + FP)

    The counting is DELEGATED to the finished implementation
    :func:`morie.fn.precn.precision`; this wrapper adds the confusion
    counts, the F1 companion and the Geron-flavoured reporting. Precision
    answers "of the instances I flagged, how many were right?" -- a
    classifier that flags a single certain positive scores 1.0, which is
    why precision is only meaningful next to recall.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels, same length.
    pos_label : scalar, default 1
        Label treated as positive.

    Returns
    -------
    result : RichResult
        Keys: precision, tp, fp, fn, f1, estimate, n, method.

    Examples
    --------
    >>> r = geron_precision([1, 0, 1, 1, 0], [1, 1, 1, 0, 0])
    >>> r["tp"], r["fp"], r["fn"]
    (2, 1, 1)
    >>> round(float(r["precision"]), 6)
    0.666667
    >>> float(geron_precision([0, 1], [0, 1])["precision"])
    1.0

    References
    ----------
    Geron Ch 3
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.size == 0:
        raise ValueError("geron_precision: y_true is empty")
    if yt.size != yp.size:
        raise ValueError(f"geron_precision: y_true has {yt.size} entries but y_pred has {yp.size}")

    res = _precision(yt, yp, pos_label=pos_label)
    tp, fp, fn = int(res.extra["tp"]), int(res.extra["fp"]), int(res.extra["fn"])
    if tp + fp == 0:
        raise ValueError(
            "geron_precision: no instance was predicted positive, so TP + FP = 0 and precision is undefined"
        )
    prec = tp / (tp + fp)
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp + fn) and (prec + rec) > 0 else 0.0
    return RichResult(
        title="Precision",
        summary_lines=[("Precision", prec), ("TP", tp), ("FP", fp)],
        interpretation="Precision alone can be gamed by predicting positive only when certain; read it with recall.",
        payload={
            "precision": prec,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": float(f1),
            "estimate": prec,
            "n": int(yt.size),
            "method": "Precision TP/(TP+FP), counts delegated to morie.fn.precn.precision",
        },
    )


def cheatsheet():
    return "hmpre: Precision = TP / (TP + FP)"
