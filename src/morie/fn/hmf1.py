# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""F1 score: harmonic mean of precision and recall."""

import numpy as np

from ._richresult import RichResult
from .hmcfm import geron_confusion_matrix

__all__ = ["geron_f1_score"]


def geron_f1_score(y_true, y_pred, pos_label=1, average="binary"):
    """
    F1 score: harmonic mean of precision and recall.

    Formula: F1 = 2*P*R / (P + R)

    Precision and recall are read off a confusion matrix, DELEGATED to
    :func:`morie.fn.hmcfm.geron_confusion_matrix` (which itself delegates
    the counting to ``grcfm``), so the three modules cannot disagree about
    the row/column convention.

    Being a harmonic mean, F1 is dragged down by whichever of precision
    and recall is worse: 0.9 and 0.1 give 0.18, not 0.5. When both are
    zero the score is defined as 0 rather than NaN, and the reason is
    recorded in ``warnings``.

    Parameters
    ----------
    y_true, y_pred : array-like
        Labels.
    pos_label : scalar, default 1
        Positive class for ``average="binary"``.
    average : {"binary", "macro", "micro", None}, default "binary"
        ``None`` returns the per-class vector as ``f1``.

    Returns
    -------
    result : RichResult
        Keys: f1, precision, recall, tp, fp, fn, per_class_f1,
        estimate, n, method.

    Examples
    --------
    Two of three predicted positives are right (P = 2/3) and two of two
    actual positives are found (R = 1), so F1 = 0.8:

    >>> r = geron_f1_score([0, 0, 1, 1], [0, 1, 1, 1])
    >>> round(r["precision"], 6), round(r["recall"], 6), round(r["f1"], 6)
    (0.666667, 1.0, 0.8)
    >>> r["tp"], r["fp"], r["fn"]
    (2, 1, 0)

    Predicting no positives at all scores 0, not NaN:

    >>> geron_f1_score([0, 1], [0, 0])["f1"]
    0.0

    Macro averaging over the two classes:

    >>> round(geron_f1_score([0, 0, 1, 1], [0, 1, 1, 1], average="macro")["f1"], 6)
    0.733333

    References
    ----------
    Géron Ch 3
    """
    if average not in ("binary", "macro", "micro", None):
        raise ValueError(f"geron_f1_score: average must be 'binary', 'macro', 'micro' or None, got {average!r}")
    cm_res = geron_confusion_matrix(y_true, y_pred)
    cm = np.asarray(cm_res["matrix"], dtype=float)
    labels = list(cm_res["labels"])
    per_class = list(cm_res["f1"])
    warn = []

    if average == "binary":
        if len(labels) != 2:
            raise ValueError(
                f"geron_f1_score: average='binary' needs exactly 2 classes, got {len(labels)} ({labels}); "
                "use average='macro' or 'micro'"
            )
        if pos_label not in labels:
            raise ValueError(f"geron_f1_score: pos_label={pos_label!r} not among the observed labels {labels}")
        k = labels.index(pos_label)
        tp = float(cm[k, k])
        fp = float(cm[:, k].sum() - tp)
        fn = float(cm[k, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if (tp + fp) == 0:
            warn.append("no positive predictions were made; precision is defined as 0")
    elif average == "macro":
        prec = float(np.nanmean(np.asarray(cm_res["precision"], dtype=float)))
        rec = float(np.nanmean(np.asarray(cm_res["recall"], dtype=float)))
        f1 = float(np.nanmean(np.asarray(per_class, dtype=float)))
        tp = float(np.trace(cm))
        fp = fn = float(cm.sum() - tp)
    elif average == "micro":
        tp = float(np.trace(cm))
        fp = fn = float(cm.sum() - tp)
        prec = rec = tp / cm.sum() if cm.sum() else 0.0
        f1 = prec
    else:
        prec = list(cm_res["precision"])
        rec = list(cm_res["recall"])
        f1 = per_class
        tp = float(np.trace(cm))
        fp = fn = float(cm.sum() - tp)

    scalar = f1 if not isinstance(f1, list) else float(np.nanmean(np.asarray(f1, dtype=float)))

    return RichResult(
        title="F1 score",
        summary_lines=[("F1", f1), ("Precision", prec), ("Recall", rec)],
        warnings=warn,
        interpretation="F1 is a harmonic mean, so it tracks the weaker of precision and recall.",
        payload={
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "per_class_f1": per_class,
            "labels": labels,
            "average": average,
            "estimate": scalar,
            "n": int(cm_res["n"]),
            "method": "F1 = 2PR/(P+R) from a confusion matrix delegated to hmcfm",
        },
    )


def cheatsheet():
    return "hmf1: F1 score: harmonic mean of precision and recall"
