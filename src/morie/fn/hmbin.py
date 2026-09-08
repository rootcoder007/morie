# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Binary classification: predict one of two classes using probability threshold."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_binary_classification"]


def geron_binary_classification(X, theta, threshold=0.5, y_true=None):
    """
    Binary classification: predict one of two classes using a probability threshold.

    Formula: y_hat = I(p_hat >= 0.5)

    The score is the logistic model p_hat = sigmoid(X theta); the decision
    rule compares it against `threshold`. When `y_true` is supplied the
    confusion counts, precision, recall and F1 are reported too.

    Parameters
    ----------
    X : array-like, shape (n, k)
        Design matrix (include your own intercept column).
    theta : array-like, shape (k,)
        Logistic coefficients.
    threshold : float, default 0.5
        Decision threshold on p_hat; must lie in [0, 1].
    y_true : array-like, optional
        Observed 0/1 labels, for the confusion summary.

    Returns
    -------
    result : RichResult
        Keys: y_pred, p_hat, logit, accuracy, precision, recall, f1,
        estimate, n, method.

    Examples
    --------
    >>> r = geron_binary_classification([[1.0, 2.0], [1.0, -3.0]], [0.5, 0.5])
    >>> [round(float(p), 6) for p in r["p_hat"]]
    [0.817574, 0.268941]
    >>> [int(v) for v in r["y_pred"]]
    [1, 0]
    >>> r2 = geron_binary_classification([[1.0, 2.0], [1.0, -3.0]], [0.5, 0.5], y_true=[1, 1])
    >>> float(r2["accuracy"]), float(r2["recall"])
    (0.5, 0.5)

    References
    ----------
    Géron Ch 3
    """
    Xm = np.asarray(X, dtype=float)
    if Xm.ndim == 1:
        Xm = Xm.reshape(1, -1)
    if Xm.ndim != 2:
        raise ValueError(f"geron_binary_classification: X must be 2-D, got ndim={Xm.ndim}")
    th = np.asarray(theta, dtype=float).ravel()
    if Xm.shape[0] == 0:
        raise ValueError("geron_binary_classification: X has no rows")
    if Xm.shape[1] != th.size:
        raise ValueError(
            f"geron_binary_classification: X has {Xm.shape[1]} columns but theta has {th.size} entries"
        )
    t = float(threshold)
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"geron_binary_classification: threshold must lie in [0, 1], got {t}")

    z = Xm @ th
    # Branch-free stable logistic: exp() is only ever applied to <= 0.
    p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
    y_pred = (p >= t).astype(int)

    acc = prec = rec = f1 = None
    tp = fp = fn = tn = None
    if y_true is not None:
        yt = np.asarray(y_true).ravel().astype(int)
        if yt.size != Xm.shape[0]:
            raise ValueError(
                f"geron_binary_classification: y_true has {yt.size} entries but X has {Xm.shape[0]} rows"
            )
        if not np.all(np.isin(yt, (0, 1))):
            raise ValueError("geron_binary_classification: y_true must contain only 0 and 1")
        tp = int(np.sum((y_pred == 1) & (yt == 1)))
        fp = int(np.sum((y_pred == 1) & (yt == 0)))
        fn = int(np.sum((y_pred == 0) & (yt == 1)))
        tn = int(np.sum((y_pred == 0) & (yt == 0)))
        acc = (tp + tn) / yt.size
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return RichResult(
        title="Binary classification",
        summary_lines=[("Threshold", t), ("Predicted positives", int(np.sum(y_pred)))],
        payload={
            "y_pred": y_pred,
            "p_hat": p,
            "logit": z,
            "threshold": t,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "estimate": float(np.mean(y_pred)),
            "n": int(Xm.shape[0]),
            "method": "Binary classification by thresholding the logistic score",
        },
    )


def cheatsheet():
    return "hmbin: Binary classification: predict one of two classes using probability threshold"
