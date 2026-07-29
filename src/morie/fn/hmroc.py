# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Receiver operating characteristic: FPR vs TPR over thresholds."""

import numpy as np

from ._richresult import RichResult
from .hmauc import geron_auc_roc

__all__ = ["geron_roc_curve"]


def geron_roc_curve(y_true, scores, pos_label=1):
    """
    Receiver operating characteristic: FPR vs TPR over thresholds.

    Formula: FPR = FP/N; TPR = TP/P

    The sweep is DELEGATED to the finished implementation
    :func:`morie.fn.hmauc.geron_auc_roc`, which already walks the
    thresholds in descending score order and returns the vertices. This
    wrapper adds the trapezoidal area (an independent route to the same
    AUC that hmauc gets from the Mann-Whitney identity), the counts at
    every vertex and Youden's J operating point.

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
        Keys: fpr, tpr, thresholds, auc, auc_trapezoid, best_threshold,
        youden_j, estimate, n, method.

    Examples
    --------
    >>> r = geron_roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> float(r["auc"])
    0.75
    >>> round(float(r["auc_trapezoid"]), 12)
    0.75
    >>> [float(v) for v in r["tpr"]]
    [0.0, 0.5, 0.5, 1.0, 1.0]
    >>> [float(v) for v in r["fpr"]]
    [0.0, 0.0, 0.5, 0.5, 1.0]

    References
    ----------
    Geron Ch 3
    """
    base = geron_auc_roc(y_true, scores, pos_label=pos_label)
    fpr = np.asarray(base["fpr"], dtype=float)
    tpr = np.asarray(base["tpr"], dtype=float)
    auc_trap = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    j = tpr - fpr
    k = int(np.argmax(j))
    return RichResult(
        title="ROC curve",
        summary_lines=[("AUC", float(base["auc"])), ("Youden J", float(j[k]))],
        interpretation="The diagonal is chance; a curve hugging the top-left separates the classes.",
        payload={
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": np.asarray(base["thresholds"], dtype=float),
            "auc": float(base["auc"]),
            "auc_trapezoid": auc_trap,
            "youden_j": float(j[k]),
            "best_threshold": float(np.asarray(base["thresholds"], dtype=float)[k]),
            "n_pos": int(base["n_pos"]),
            "n_neg": int(base["n_neg"]),
            "estimate": float(base["auc"]),
            "n": int(np.asarray(y_true).size),
            "method": "ROC sweep delegated to morie.fn.hmauc.geron_auc_roc, area re-checked by trapezoid",
        },
    )


def cheatsheet():
    return "hmroc: Receiver operating characteristic: FPR vs TPR over thresholds"
