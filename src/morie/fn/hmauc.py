# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Area under the ROC curve."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_auc_roc"]


def geron_auc_roc(y_true, scores, pos_label=1):
    """
    Area under the ROC curve.

    Formula: AUC = integral_0^1 TPR(FPR) dFPR

    Computed exactly by the Mann-Whitney identity: the AUC equals the
    probability that a random positive outscores a random negative, with
    ties counted as one half. The ROC vertices are returned as well, and
    trapezoidal integration over them reproduces the same number.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    scores : array-like
        Decision scores; larger means "more positive".
    pos_label : scalar, default 1
        Label treated as the positive class.

    Returns
    -------
    result : RichResult
        Keys: auc, fpr, tpr, thresholds, n_pos, n_neg, estimate, n, method.

    Examples
    --------
    >>> r = geron_auc_roc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> float(r["auc"])
    0.75
    >>> float(geron_auc_roc([0, 1], [0.0, 1.0])["auc"])
    1.0
    >>> float(geron_auc_roc([0, 1], [1.0, 0.0])["auc"])
    0.0
    >>> float(geron_auc_roc([0, 1], [0.5, 0.5])["auc"])
    0.5

    References
    ----------
    Géron Ch 3
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if y.size == 0:
        raise ValueError("geron_auc_roc: y_true is empty")
    if y.size != s.size:
        raise ValueError(f"geron_auc_roc: y_true has {y.size} entries but scores has {s.size}")
    if not np.all(np.isfinite(s)):
        raise ValueError("geron_auc_roc: scores contain non-finite values")

    pos = y == pos_label
    n_pos = int(np.sum(pos))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"geron_auc_roc: ROC AUC needs both classes present (got {n_pos} positive, {n_neg} negative)"
        )

    # Mann-Whitney U via mid-ranks, which credits ties with 0.5.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    auc = (float(np.sum(ranks[pos])) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    # ROC vertices, descending threshold.
    desc = np.argsort(-s, kind="mergesort")
    tp = fp = 0
    fpr = [0.0]
    tpr = [0.0]
    thr = [np.inf]
    k = 0
    sd = s[desc]
    pd = pos[desc]
    while k < s.size:
        j = k
        while j + 1 < s.size and sd[j + 1] == sd[k]:
            j += 1
        tp += int(np.sum(pd[k : j + 1]))
        fp += int(np.sum(~pd[k : j + 1]))
        fpr.append(fp / n_neg)
        tpr.append(tp / n_pos)
        thr.append(float(sd[k]))
        k = j + 1

    return RichResult(
        title="ROC AUC",
        summary_lines=[("AUC", auc), ("Positives", n_pos), ("Negatives", n_neg)],
        interpretation=("AUC 0.5 is chance; 1.0 is perfect separation; below 0.5 means the score is inverted."),
        payload={
            "auc": auc,
            "fpr": np.asarray(fpr, dtype=float),
            "tpr": np.asarray(tpr, dtype=float),
            "thresholds": np.asarray(thr, dtype=float),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "estimate": auc,
            "n": int(y.size),
            "method": "ROC AUC via the Mann-Whitney identity (ties credited 0.5)",
        },
    )


def cheatsheet():
    return "hmauc: Area under the ROC curve"
