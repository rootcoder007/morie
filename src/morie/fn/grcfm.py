# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Confusion matrix for binary/multiclass classification."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_confusion_matrix"]

_METHOD = "Confusion matrix"


def geron_confusion_matrix(y_true, y_pred, n_classes=None):
    r"""Count every (true, predicted) pair.

    .. math::
        \mathrm{CM}[i, j] = \#\{k : y_k = i \wedge \hat y_k = j\}

    Rows are truth, columns are prediction -- so row sums are class
    supports and the diagonal holds the hits.  Per-class precision,
    recall and F1 are derived here rather than left to the caller,
    because reading them off the matrix by eye is where the row/column
    convention gets flipped.

    Parameters
    ----------
    y_true, y_pred : array-like of int
        Class labels in ``0 .. n_classes-1``.
    n_classes : int, optional
        Number of classes; inferred as ``max(label) + 1`` if omitted.

    Returns
    -------
    RichResult
        Payload keys ``matrix``, ``accuracy``, ``precision``,
        ``recall``, ``f1``, ``support``, ``macro_f1``, ``estimate``
        (accuracy), ``n``, ``method``.

    References
    ----------
    Géron Ch 3, Confusion Matrix section.

    Examples
    --------
    >>> r = geron_confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1])
    >>> r["matrix"]
    [[1, 1], [0, 2]]
    >>> round(r["accuracy"], 6)
    0.75
    >>> [round(p, 6) for p in r["precision"]]
    [1.0, 0.666667]
    >>> [round(p, 6) for p in r["recall"]]
    [0.5, 1.0]
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size != y_pred.size:
        raise ValueError(
            f"y_true and y_pred must have equal length, got {y_true.size} and {y_pred.size}."
        )
    if y_true.size == 0:
        raise ValueError("no observations supplied.")
    try:
        yt = y_true.astype(int)
        yp = y_pred.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be integer class indices.") from exc
    if not np.array_equal(yt, y_true) or not np.array_equal(yp, y_pred):
        raise ValueError("labels must be whole numbers used as class indices.")
    if yt.min() < 0 or yp.min() < 0:
        raise ValueError("class indices must be non-negative.")
    K = int(max(yt.max(), yp.max())) + 1 if n_classes is None else int(n_classes)
    if K < 1:
        raise ValueError(f"n_classes must be at least 1, got {K}.")
    if yt.max() >= K or yp.max() >= K:
        raise ValueError(
            f"labels reach {int(max(yt.max(), yp.max()))} but n_classes={K}."
        )

    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (yt, yp), 1)

    tp = np.diag(cm).astype(float)
    pred_tot = cm.sum(axis=0).astype(float)
    true_tot = cm.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(pred_tot > 0, tp / pred_tot, np.nan)
        rec = np.where(true_tot > 0, tp / true_tot, np.nan)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    acc = float(tp.sum() / cm.sum())
    macro = float(np.nanmean(f1)) if np.any(np.isfinite(f1)) else float("nan")

    return RichResult(
        title="Confusion matrix",
        summary_lines=[("Accuracy", acc), ("Classes", K)],
        tables=[{
            "title": "rows = true, columns = predicted",
            "headers": [f"pred {j}" for j in range(K)],
            "rows": cm.tolist(),
        }],
        payload={
            "matrix": cm.tolist(),
            "accuracy": acc,
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "f1": f1.tolist(),
            "support": true_tot.astype(int).tolist(),
            "macro_f1": macro,
            "n_classes": K,
            "estimate": acc,
            "n": int(y_true.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grcfm: confusion matrix CM[i,j] = count(true i, pred j), with precision/recall/F1"
