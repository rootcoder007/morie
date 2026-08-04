# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""F1 score -- the harmonic mean of precision and recall."""

from . import _array_core as np

from ._richresult import RichResult
from .grcfm import geron_confusion_matrix

__all__ = ["geron_f1_score"]

_METHOD = "F1 score (Eq 3-3)"


def geron_f1_score(y_true, y_pred, positive_class=1):
    r"""Géron Eq 3-3.

    .. math::
        F_1 = \frac{2\,\text{precision}\times\text{recall}}
                   {\text{precision} + \text{recall}}

    The harmonic mean, not the arithmetic one: it is dragged down by
    whichever of the two is worse.  A classifier with precision 1.0 and
    recall 0.02 scores 0.51 on the arithmetic mean and 0.039 on F1, and
    the second number is the honest one.

    Counting is delegated to
    :func:`morie.fn.grcfm.geron_confusion_matrix`, which already
    derives per-class precision, recall and F1 from the matrix; this
    function selects the class of interest and reports the macro
    average alongside.

    Parameters
    ----------
    y_true, y_pred : array-like of int, shape (m,)
    positive_class : int, optional
        Class whose F1 is reported as ``estimate``. Default 1.

    Returns
    -------
    RichResult
        Payload keys ``f1``, ``precision``, ``recall``, ``macro_f1``,
        ``per_class_f1``, ``confusion_matrix``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 3, Eq 3-3 (F1 score).

    Examples
    --------
    Three positives, two found, one false alarm: precision 2/3,
    recall 2/3, F1 2/3:

    >>> r = geron_f1_score([1, 1, 1, 0], [1, 1, 0, 1])
    >>> round(r["precision"], 10), round(r["recall"], 10)
    (0.6666666667, 0.6666666667)
    >>> round(r["f1"], 10)
    0.6666666667

    The harmonic mean punishes imbalance -- precision 1, recall 1/3
    gives 0.5, not 0.667:

    >>> r2 = geron_f1_score([1, 1, 1, 0], [1, 0, 0, 0])
    >>> (r2["precision"], round(r2["recall"], 10), r2["f1"])
    (1.0, 0.3333333333, 0.5)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size != y_pred.size:
        raise ValueError(f"y_true has {y_true.size} entries but y_pred has {y_pred.size}.")
    if y_true.size == 0:
        raise ValueError("F1 over zero instances is undefined.")
    labels = np.unique(np.concatenate([y_true, y_pred])).astype(int)
    if np.any(labels < 0):
        raise ValueError(f"labels must be non-negative integers, got {labels.tolist()}.")
    pc = int(positive_class)
    if pc not in labels.tolist():
        raise ValueError(
            f"positive_class={pc} appears in neither y_true nor y_pred "
            f"(labels present: {labels.tolist()}); F1 for it is undefined."
        )

    cm = geron_confusion_matrix(y_true, y_pred)   # classes indexed 0 .. max(label)
    prec = float(np.asarray(cm["precision"])[pc])
    rec = float(np.asarray(cm["recall"])[pc])
    f1 = 0.0 if (not np.isfinite(prec) or not np.isfinite(rec) or prec + rec == 0) \
        else 2.0 * prec * rec / (prec + rec)

    warns = []
    if not np.isfinite(prec):
        warns.append(f"class {pc} was never predicted, so precision is undefined; F1 reported as 0.")
    if not np.isfinite(rec):
        warns.append(f"class {pc} never occurs in y_true, so recall is undefined; F1 reported as 0.")

    return RichResult(
        title="F1 score",
        summary_lines=[("F1", f1), ("Precision", prec), ("Recall", rec)],
        warnings=warns,
        payload={
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "macro_f1": cm["macro_f1"],
            "per_class_f1": cm["f1"],
            "confusion_matrix": cm["matrix"],
            "positive_class": pc,
            "estimate": f1,
            "n": int(y_true.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grf1: F1 = 2PR/(P+R), harmonic mean; counts delegated to grcfm"


# compact alias per ledger/NAMING.md
geronf1score = geron_f1_score
