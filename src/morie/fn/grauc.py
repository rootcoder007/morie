# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Area under the ROC curve."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_auc_roc"]

_METHOD = "Area under the ROC curve (trapezoid rule)"


def geron_auc_roc(y_true, y_scores, pos_label=1):
    r"""ROC curve and its area, by the trapezoid rule over all thresholds.

    Scores are sorted descending and every distinct score becomes a
    threshold; the cumulative true- and false-positive counts give the
    ROC points, and

    .. math::
        \text{AUC} = \int_0^1 \text{TPR}\,d(\text{FPR})

    is the trapezoid sum over those points.  With ties handled this way
    the result equals the Mann-Whitney statistic -- the probability that
    a random positive outranks a random negative, counting ties as half.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_scores : array-like
        Decision scores; higher means more positive.
    pos_label : optional
        Value of ``y_true`` treated as the positive class.

    Returns
    -------
    RichResult
        Payload keys ``auc``, ``fpr``, ``tpr``, ``thresholds``,
        ``n_pos``, ``n_neg``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 3, AUC-ROC section.

    Examples
    --------
    >>> r = geron_auc_roc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> round(r["auc"], 6)
    0.75
    >>> r["fpr"][0], r["tpr"][0]
    (0.0, 0.0)
    >>> r["fpr"][-1], r["tpr"][-1]
    (1.0, 1.0)

    A perfect ranking scores 1.0, a reversed one 0.0:

    >>> round(float(geron_auc_roc([0, 0, 1, 1], [0.0, 0.1, 0.2, 0.3])), 6)
    1.0
    >>> round(float(geron_auc_roc([0, 0, 1, 1], [0.3, 0.2, 0.1, 0.0])), 6)
    0.0
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()
    if y_true.size != y_scores.size:
        raise ValueError(
            f"y_true and y_scores must have equal length, got {y_true.size} and {y_scores.size}."
        )
    if y_true.size == 0:
        raise ValueError("no observations supplied.")
    if not np.all(np.isfinite(y_scores)):
        raise ValueError("y_scores contains non-finite values.")
    pos = (y_true == pos_label)
    n_pos = int(pos.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"ROC needs both classes present; got {n_pos} positive and "
            f"{n_neg} negative observations."
        )

    order = np.argsort(-y_scores, kind="mergesort")
    s = y_scores[order]
    p = pos[order].astype(float)

    tp = np.cumsum(p)
    fp = np.cumsum(1.0 - p)
    # Keep only the last index of each run of equal scores -- a threshold
    # cannot split tied scores.
    keep = np.r_[np.diff(s) != 0, True]
    tp = np.r_[0.0, tp[keep]]
    fp = np.r_[0.0, fp[keep]]
    thresholds = np.r_[np.inf, s[keep]]

    tpr = tp / n_pos
    fpr = fp / n_neg
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))

    return RichResult(
        title="ROC AUC",
        summary_lines=[("AUC", auc), ("Positives", n_pos), ("Negatives", n_neg)],
        interpretation=(
            "AUC is the probability a random positive is ranked above a random "
            "negative; 0.5 is chance."
        ),
        payload={
            "auc": auc,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "estimate": auc,
            "n": int(y_true.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grauc: ROC curve + AUC by trapezoid rule over all thresholds"
