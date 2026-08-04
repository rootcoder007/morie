# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Precision = TP / (TP + FP)."""

from . import _array_core as np

from ._richresult import RichResult
from .grcfm import geron_confusion_matrix

__all__ = ["geron_precision"]

_METHOD = "Precision TP/(TP+FP)"


def geron_precision(y_true, y_pred, positive=1, average=None):
    r"""Share of positive predictions that were right.

    .. math::
        \mathrm{precision} = \frac{TP}{TP + FP}

    The counting is delegated to :func:`morie.fn.grcfm.geron_confusion_matrix`
    -- this module only picks the class and handles the degenerate case.
    If the classifier never predicts the positive class the denominator is
    zero and precision is *undefined*; that raises rather than silently
    returning 0, because "predicted nothing" and "predicted nothing right"
    are different failures.

    Parameters
    ----------
    y_true, y_pred : array-like of int
    positive : int, optional
        Class treated as positive (default 1).
    average : {None, "macro"}, optional
        ``"macro"`` returns the unweighted mean over classes that were
        predicted at least once.

    Returns
    -------
    RichResult
        Payload keys ``precision``, ``tp``, ``fp``, ``per_class``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 3, Eq 3-1 (Precision).

    Examples
    --------
    Three positive predictions, two correct:

    >>> yt = [1, 1, 0, 0, 1]
    >>> yp = [1, 0, 1, 0, 1]
    >>> r = geron_precision(yt, yp)
    >>> r["tp"], r["fp"]
    (2, 1)
    >>> round(r["precision"], 6)
    0.666667
    """
    yt = np.asarray(y_true).ravel().astype(int)
    yp = np.asarray(y_pred).ravel().astype(int)
    if yt.size == 0:
        raise ValueError("y_true is empty.")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true has {yt.size} labels but y_pred has {yp.size}.")
    n_classes = int(max(yt.max(), yp.max())) + 1
    cm = geron_confusion_matrix(yt, yp, n_classes=n_classes)
    per_class = np.asarray(cm["precision"], dtype=float)
    M = np.asarray(cm["matrix"], dtype=float)
    predicted = M.sum(axis=0)

    if average is None:
        pos = int(positive)
        if not (0 <= pos < n_classes):
            raise ValueError(f"positive class {pos} not in [0, {n_classes - 1}].")
        if predicted[pos] == 0:
            raise ValueError(
                f"class {pos} was never predicted, so TP+FP = 0 and precision is undefined."
            )
        tp = int(M[pos, pos])
        fp = int(predicted[pos] - M[pos, pos])
        val = tp / (tp + fp)
    elif average == "macro":
        seen = predicted > 0
        if not seen.any():
            raise ValueError("no class was ever predicted; macro precision is undefined.")
        val = float(per_class[seen].mean())
        tp = int(np.trace(M))
        fp = int(M.sum() - np.trace(M))
    else:
        raise ValueError(f"average must be None or 'macro', got {average!r}.")

    return RichResult(
        title="Precision",
        summary_lines=[("Precision", float(val)), ("TP", tp), ("FP", fp)],
        payload={
            "precision": float(val),
            "tp": tp,
            "fp": fp,
            "per_class": per_class.tolist(),
            "estimate": float(val),
            "n": int(yt.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grpre: precision = TP/(TP+FP); counts delegated to grcfm; raises when the class is never predicted"


# compact alias per ledger/NAMING.md
geronprecision = geron_precision
