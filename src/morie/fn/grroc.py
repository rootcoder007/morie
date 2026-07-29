# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Receiver operating characteristic curve (TPR vs FPR over thresholds)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_roc_curve"]

_METHOD = "ROC curve"


def _sorted_counts(y_true, y_scores):
    yt = np.asarray(y_true).ravel()
    s = np.asarray(y_scores, dtype=float).ravel()
    if yt.size == 0:
        raise ValueError("y_true is empty.")
    if yt.shape != s.shape:
        raise ValueError(f"y_true has {yt.size} labels but y_scores has {s.size}.")
    if not np.all(np.isfinite(s)):
        raise ValueError("y_scores contains non-finite values.")
    uniq = set(np.unique(yt).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"y_true must be binary 0/1, got labels {sorted(uniq)}.")
    P = int((yt == 1).sum())
    N = int((yt == 0).sum())
    if P == 0 or N == 0:
        raise ValueError(
            f"need both classes present: got {P} positives and {N} negatives."
        )
    order = np.argsort(-s, kind="mergesort")
    return yt[order].astype(int), s[order], P, N


def geron_roc_curve(y_true, y_scores):
    r"""True-positive rate against false-positive rate at every threshold.

    .. math::
        \mathrm{TPR}(t) = \frac{TP(t)}{TP(t) + FN(t)}, \qquad
        \mathrm{FPR}(t) = \frac{FP(t)}{FP(t) + TN(t)}

    Thresholds are the distinct scores, walked from high to low, so the
    curve is exact rather than sampled on a grid.  Tied scores are
    collapsed into a single point -- otherwise the curve would show
    staircase segments that no threshold can actually realise.  AUC is
    the trapezoid area (see also :mod:`morie.fn.grauc`).

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_scores : array-like of float
        Higher = more positive.

    Returns
    -------
    RichResult
        Payload keys ``fpr``, ``tpr``, ``thresholds``, ``auc``,
        ``estimate`` (auc), ``n``, ``method``.

    References
    ----------
    Géron Ch 3, ROC Curve section.

    Examples
    --------
    A perfect ranking reaches the top-left corner and has AUC 1:

    >>> r = geron_roc_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    >>> r["auc"]
    1.0
    >>> r["tpr"]
    [0.0, 0.5, 1.0, 1.0, 1.0]

    One swapped pair out of four positive-negative pairs costs 0.25:

    >>> geron_roc_curve([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4])["auc"]
    0.75
    """
    ys, ss, P, N = _sorted_counts(y_true, y_scores)
    tp = fp = 0
    fpr = [0.0]
    tpr = [0.0]
    thr = [float(np.inf)]
    i = 0
    while i < ys.size:
        j = i
        while j + 1 < ys.size and ss[j + 1] == ss[i]:
            j += 1
        tp += int((ys[i : j + 1] == 1).sum())
        fp += int((ys[i : j + 1] == 0).sum())
        fpr.append(fp / N)
        tpr.append(tp / P)
        thr.append(float(ss[i]))
        i = j + 1
    # Trapezoid rule written out: np.trapz was removed in numpy 2.
    auc = float(
        sum(
            (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2.0
            for i in range(len(fpr) - 1)
        )
    )

    return RichResult(
        title="ROC curve",
        summary_lines=[("Points", len(fpr)), ("AUC", auc)],
        payload={
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thr,
            "auc": auc,
            "estimate": auc,
            "n": int(ys.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grroc: TPR vs FPR over the distinct scores (ties collapsed); AUC by trapezoid"
