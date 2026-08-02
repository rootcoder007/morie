# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Confusion matrix: rows = actual, columns = predicted classes."""

from . import _array_core as np

from ._richresult import RichResult
from .grcfm import geron_confusion_matrix as _grcfm

__all__ = ["geron_confusion_matrix"]


def geron_confusion_matrix(y_true, y_pred, n_classes=None, labels=None):
    """
    Confusion matrix: rows = actual, columns = predicted classes.

    Formula: C[i,j] = #{predicted class j | actual class i}

    The counting and the per-class precision/recall/F1 are DELEGATED to
    :func:`morie.fn.grcfm.geron_confusion_matrix`, which already
    implements exactly this formula; this module adds label handling for
    non-integer classes (strings are mapped to indices in sorted order and
    returned as ``labels``) and the row/column marginals.

    Parameters
    ----------
    y_true, y_pred : array-like
        Class labels; any dtype. Non-integer labels are encoded by sorted
        order across both arrays.
    n_classes : int, optional
        Force the matrix size (integer labels only).
    labels : sequence, optional
        Explicit label ordering; every observed label must appear in it.

    Returns
    -------
    result : RichResult
        Keys: matrix, labels, accuracy, precision, recall, f1, support,
        predicted_totals, macro_f1, estimate, n, method.

    Examples
    --------
    >>> r = geron_confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1])
    >>> r["matrix"]
    [[1, 1], [0, 2]]
    >>> round(r["accuracy"], 6)
    0.75
    >>> r["predicted_totals"]
    [1, 3]

    String labels are encoded in sorted order:

    >>> r2 = geron_confusion_matrix(["cat", "dog", "cat"], ["cat", "cat", "cat"])
    >>> r2["labels"]
    ['cat', 'dog']
    >>> r2["matrix"]
    [[2, 0], [1, 0]]

    References
    ----------
    Géron Ch 3
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.size != yp.size:
        raise ValueError(f"geron_confusion_matrix: y_true has {yt.size} entries but y_pred has {yp.size}")
    if yt.size == 0:
        raise ValueError("geron_confusion_matrix: no observations supplied")

    if labels is not None:
        lab = list(labels)
        if len(set(map(str, lab))) != len(lab):
            raise ValueError("geron_confusion_matrix: labels contains duplicates")
        index = {v: i for i, v in enumerate(lab)}
        missing = [v for v in set(yt.tolist()) | set(yp.tolist()) if v not in index]
        if missing:
            raise ValueError(f"geron_confusion_matrix: observed labels missing from `labels`: {sorted(map(str, missing))}")
        it = np.array([index[v] for v in yt.tolist()], dtype=int)
        ip = np.array([index[v] for v in yp.tolist()], dtype=int)
        K = len(lab)
    elif np.issubdtype(yt.dtype, np.integer) and np.issubdtype(yp.dtype, np.integer):
        it, ip = yt.astype(int), yp.astype(int)
        if it.min() < 0 or ip.min() < 0:
            raise ValueError("geron_confusion_matrix: integer class indices must be non-negative")
        K = int(max(it.max(), ip.max())) + 1 if n_classes is None else int(n_classes)
        lab = list(range(K))
    else:
        lab = sorted(set(yt.tolist()) | set(yp.tolist()), key=str)
        index = {v: i for i, v in enumerate(lab)}
        it = np.array([index[v] for v in yt.tolist()], dtype=int)
        ip = np.array([index[v] for v in yp.tolist()], dtype=int)
        K = len(lab)

    base = _grcfm(it, ip, n_classes=K)
    cm = np.asarray(base["matrix"], dtype=int)

    return RichResult(
        title="Confusion matrix",
        summary_lines=[("Accuracy", float(base["accuracy"])), ("Classes", int(K))],
        tables=[{"title": "rows = actual, columns = predicted", "headers": [f"pred {v}" for v in lab], "rows": cm.tolist()}],
        payload={
            "matrix": cm.tolist(),
            "labels": lab,
            "accuracy": float(base["accuracy"]),
            "precision": list(base["precision"]),
            "recall": list(base["recall"]),
            "f1": list(base["f1"]),
            "support": list(base["support"]),
            "predicted_totals": cm.sum(axis=0).astype(int).tolist(),
            "macro_f1": float(base["macro_f1"]),
            "n_classes": int(K),
            "estimate": float(base["accuracy"]),
            "n": int(yt.size),
            "method": "confusion matrix C[i,j] = count(actual i, predicted j); counting delegated to grcfm",
        },
    )


def cheatsheet():
    return "hmcfm: Confusion matrix: rows = actual, columns = predicted classes"
