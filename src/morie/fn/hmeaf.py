# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Error analysis via normalized confusion matrix row/column inspection."""

from . import _array_core as np

from ._richresult import RichResult
from .hmcfm import geron_confusion_matrix

__all__ = ["geron_error_analysis"]


def geron_error_analysis(y_true, y_pred, top_k=5):
    """
    Error analysis via normalized confusion matrix row/column inspection.

    Formula: E_norm[i,j] = C[i,j] / sum_j C[i,j] - diag correction

    The counting is DELEGATED to
    :func:`morie.fn.hmcfm.geron_confusion_matrix`. What this module adds is
    the step Géron actually recommends: divide each row by its support so
    classes with different frequencies are comparable, then zero the
    diagonal so the plot shows errors rather than the (always dominant)
    correct predictions. The remaining mass is ranked, giving the worst
    confusions directly.

    Row-normalised entries answer "when the truth is i, how often do we
    say j"; the column-normalised view answers the reverse, and both are
    returned because the asymmetry is usually where the diagnosis is.

    Parameters
    ----------
    y_true, y_pred : array-like
        Labels.
    top_k : int, default 5
        How many confusions to list.

    Returns
    -------
    result : RichResult
        Keys: normalized, error_matrix, column_normalized, top_confusions,
        per_class_error_rate, worst_class, error_rate, estimate, n, method.

    Examples
    --------
    Class 0 is misread as class 1 half the time; class 1 is never wrong:

    >>> r = geron_error_analysis([0, 0, 1, 1], [0, 1, 1, 1])
    >>> [[round(v, 6) for v in row] for row in r["normalized"]]
    [[0.5, 0.5], [0.0, 1.0]]
    >>> [[round(v, 6) for v in row] for row in r["error_matrix"]]
    [[0.0, 0.5], [0.0, 0.0]]
    >>> r["top_confusions"][0]
    (0, 1, 0.5)
    >>> round(r["error_rate"], 6)
    0.25
    >>> r["worst_class"]
    0

    References
    ----------
    Géron Ch 3
    """
    k = int(top_k)
    if k < 1:
        raise ValueError(f"geron_error_analysis: top_k must be >= 1, got {top_k!r}")
    base = geron_confusion_matrix(y_true, y_pred)
    cm = np.asarray(base["matrix"], dtype=float)
    labels = list(base["labels"])

    row_tot = cm.sum(axis=1)
    col_tot = cm.sum(axis=0)
    if np.any(row_tot == 0):
        empty = [labels[i] for i in np.flatnonzero(row_tot == 0).tolist()]
        raise ValueError(f"geron_error_analysis: classes {empty} never occur in y_true; row normalisation is undefined")
    norm = cm / row_tot[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        colnorm = np.where(col_tot[None, :] > 0, cm / np.where(col_tot == 0, 1.0, col_tot)[None, :], 0.0)
    err = norm.copy()
    np.fill_diagonal(err, 0.0)

    pairs = [(i, j, float(err[i, j])) for i in range(err.shape[0]) for j in range(err.shape[1]) if err[i, j] > 0]
    pairs.sort(key=lambda t: (-t[2], t[0], t[1]))
    top = [(labels[i], labels[j], v) for i, j, v in pairs[:k]]
    per_class_err = err.sum(axis=1)
    worst = int(np.argmax(per_class_err))

    return RichResult(
        title="Error analysis",
        summary_lines=[("Error rate", float(1.0 - base["accuracy"])), ("Worst class", labels[worst])],
        tables=[{"title": "row-normalised errors (diagonal zeroed)", "headers": [f"pred {v}" for v in labels], "rows": err.round(6).tolist()}],
        interpretation="Rows are normalised by support, so a rare class's mistakes are not hidden by a common class's volume.",
        payload={
            "normalized": norm.tolist(),
            "error_matrix": err.tolist(),
            "column_normalized": colnorm.tolist(),
            "top_confusions": top,
            "per_class_error_rate": per_class_err.tolist(),
            "worst_class": labels[worst],
            "labels": labels,
            "error_rate": float(1.0 - base["accuracy"]),
            "accuracy": float(base["accuracy"]),
            "estimate": float(1.0 - base["accuracy"]),
            "n": int(base["n"]),
            "method": "row-normalised confusion matrix with the diagonal removed; counting delegated to hmcfm",
        },
    )


def cheatsheet():
    return "hmeaf: Error analysis via normalized confusion matrix row/column inspection"
