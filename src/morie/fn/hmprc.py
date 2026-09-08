# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Precision-recall curve over decision thresholds."""

from . import _array_core as np

from ._richresult import RichResult
from .prcpl import precision_recall_curve as _prc

__all__ = ["geron_precision_recall_curve"]


def geron_precision_recall_curve(y_true, scores, pos_label=1):
    """
    Precision-recall curve over decision thresholds.

    Formula: {(P(t), R(t)) : t in thresholds}

    The curve itself is DELEGATED to the finished implementation
    :func:`morie.fn.prcpl.precision_recall_curve`, which sweeps every
    threshold in descending score order and integrates the average
    precision. This wrapper adds the thresholds, the best-F1 operating
    point and the highest recall still attainable at 90 % precision --
    the question Geron actually asks of the curve.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    scores : array-like
        Decision scores, larger meaning more positive.
    pos_label : scalar, default 1
        Label treated as positive.

    Returns
    -------
    result : RichResult
        Keys: precision, recall, thresholds, average_precision, best_f1,
        best_threshold, recall_at_90_precision, estimate, n, method.

    Examples
    --------
    A perfectly ordered pair gives average precision 1:

    >>> r = geron_precision_recall_curve([0, 1], [0.1, 0.9])
    >>> float(r["average_precision"])
    1.0

    With the order reversed the single positive is only reached after one
    false positive, so precision there is 1/2:

    >>> r2 = geron_precision_recall_curve([1, 0], [0.1, 0.9])
    >>> float(r2["average_precision"])
    0.5
    >>> round(float(r2["best_f1"]), 6)
    0.666667

    References
    ----------
    Geron Ch 3
    """
    yt = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if yt.size == 0:
        raise ValueError("geron_precision_recall_curve: y_true is empty")
    if yt.size != s.size:
        raise ValueError(f"geron_precision_recall_curve: y_true has {yt.size} entries but scores has {s.size}")
    if not np.all(np.isfinite(s)):
        raise ValueError("geron_precision_recall_curve: scores contain non-finite values")
    bin_y = (yt == pos_label).astype(int)
    if bin_y.sum() == 0:
        raise ValueError("geron_precision_recall_curve: no positive instance, the curve is undefined")

    res = _prc(bin_y, s)
    prec = np.asarray(res.extra["precision"], dtype=float)
    rec = np.asarray(res.extra["recall"], dtype=float)
    ap = float(res.extra["average_precision"])

    # Threshold that produced each (P, R) vertex; the leading (1, 0) point
    # corresponds to an infinite threshold at which nothing is flagged.
    thr = np.concatenate(([np.inf], np.sort(s)[::-1]))

    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / np.where(denom > 0, denom, 1.0), 0.0)
    k = int(np.argmax(f1))
    ok = prec >= 0.9
    rec90 = float(np.max(rec[ok])) if np.any(ok) else 0.0

    return RichResult(
        title="Precision-recall curve",
        summary_lines=[("Average precision", ap), ("Best F1", float(f1[k])), ("Recall at P>=0.9", rec90)],
        interpretation="Precision drops as the threshold falls; pick the threshold from the precision you must hold.",
        payload={
            "precision": prec,
            "recall": rec,
            "thresholds": thr,
            "average_precision": ap,
            "f1": f1,
            "best_f1": float(f1[k]),
            "best_threshold": float(thr[k]),
            "recall_at_90_precision": rec90,
            "estimate": ap,
            "n": int(yt.size),
            "method": "PR curve delegated to morie.fn.prcpl.precision_recall_curve",
        },
    )


def cheatsheet():
    return "hmprc: Precision-recall curve over decision thresholds"
