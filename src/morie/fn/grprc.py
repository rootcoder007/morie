# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Precision-recall curve: precision and recall over thresholds."""

import numpy as np

from ._richresult import RichResult
from .grroc import _sorted_counts

__all__ = ["geron_precision_recall_curve"]

_METHOD = "Precision-recall curve"


def geron_precision_recall_curve(y_true, y_scores):
    r"""Precision and recall at every distinct decision threshold.

    .. math::
        \mathrm{precision}(t) = \frac{TP(t)}{TP(t)+FP(t)}, \qquad
        \mathrm{recall}(t) = \frac{TP(t)}{TP(t)+FN(t)}

    Recall is monotone non-increasing as the threshold rises; precision
    is *not* monotone, which is exactly why Géron plots both against the
    threshold rather than trusting one of them alone.  Average precision
    is the recall-weighted sum
    :math:`\sum_k (R_k - R_{k-1}) P_k`, which -- unlike a trapezoid --
    does not interpolate across a jump the classifier cannot reach.
    Ordering and validation are shared with :mod:`morie.fn.grroc`.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_scores : array-like of float

    Returns
    -------
    RichResult
        Payload keys ``precision``, ``recall``, ``thresholds``,
        ``average_precision``, ``best_f1``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 3, Precision/Recall curve section.

    Examples
    --------
    Perfect ranking: precision stays 1 until recall hits 1.

    >>> r = geron_precision_recall_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    >>> r["precision"]
    [1.0, 1.0, 0.6666666666666666, 0.5]
    >>> r["recall"]
    [0.5, 1.0, 1.0, 1.0]
    >>> r["average_precision"]
    1.0
    """
    ys, ss, P, N = _sorted_counts(y_true, y_scores)
    tp = fp = 0
    prec, rec, thr = [], [], []
    i = 0
    while i < ys.size:
        j = i
        while j + 1 < ys.size and ss[j + 1] == ss[i]:
            j += 1
        tp += int((ys[i : j + 1] == 1).sum())
        fp += int((ys[i : j + 1] == 0).sum())
        prec.append(tp / (tp + fp))
        rec.append(tp / P)
        thr.append(float(ss[i]))
        i = j + 1

    ap = 0.0
    prev_r = 0.0
    for p, r in zip(prec, rec):
        ap += (r - prev_r) * p
        prev_r = r
    f1 = [0.0 if (p + r) == 0 else 2 * p * r / (p + r) for p, r in zip(prec, rec)]
    best = int(np.argmax(f1))

    return RichResult(
        title="Precision-recall curve",
        summary_lines=[("Points", len(prec)), ("Average precision", float(ap)),
                       ("Best F1", float(f1[best]))],
        payload={
            "precision": prec,
            "recall": rec,
            "thresholds": thr,
            "f1": f1,
            "average_precision": float(ap),
            "best_f1": float(f1[best]),
            "best_threshold": thr[best],
            "estimate": float(ap),
            "n": int(ys.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grprc: precision/recall at each distinct score; AP = sum (R_k - R_{k-1}) P_k, no interpolation"
