# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multilabel classification: independent binary decisions per label."""

import numpy as np

from ._richresult import RichResult
from .grf1 import geron_f1_score

__all__ = ["geron_multilabel_classification"]

_METHOD = "Multilabel thresholded prediction with per-label F1"


def geron_multilabel_classification(X, Y, thresholds=0.5):
    r"""Threshold a score matrix into a label matrix and score it.

    .. math::
        \hat y_k = \mathbf 1\{\,\mathrm{score}_k(x) > t_k\,\}

    Each label gets its own decision and its own threshold: labels are
    *not* mutually exclusive, so there is no softmax and no argmax
    anywhere in this function.  A row may fire on all labels or none.

    Per-label F1 is delegated to :func:`morie.fn.grf1.geron_f1_score`.
    Both averages are reported because they answer different questions:
    macro F1 weights every label equally, micro F1 weights every
    *instance-label decision* equally and is therefore dominated by the
    common labels.

    Parameters
    ----------
    X : array-like, shape (m, K)
        Per-label scores.
    Y : array-like, shape (m, K)
        True labels, 0 or 1.
    thresholds : float or array-like, shape (K,), optional
        Per-label cut-offs, default 0.5. The comparison is strict
        (``score > t``).

    Returns
    -------
    RichResult
        Payload keys ``predictions``, ``per_label_f1``,
        ``per_label_precision``, ``per_label_recall``, ``macro_f1``,
        ``micro_f1``, ``exact_match_ratio``, ``hamming_loss``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 3, Multilabel Classification section.

    Examples
    --------
    Two labels, two instances.  Label 0 is predicted perfectly, label 1
    is missed once:

    >>> S = [[0.9, 0.9], [0.9, 0.1]]
    >>> Y = [[1, 1], [1, 1]]
    >>> r = geron_multilabel_classification(S, Y)
    >>> r["predictions"]
    [[1, 1], [1, 0]]
    >>> r["per_label_f1"]
    [1.0, 0.6666666666666666]
    >>> r["exact_match_ratio"]
    0.5

    Raising label 1's threshold above its score removes that prediction
    entirely -- thresholds are per label, so label 0 is untouched:

    >>> r2 = geron_multilabel_classification(S, Y, thresholds=[0.5, 0.95])
    >>> r2["predictions"]
    [[1, 0], [1, 0]]
    """
    S = np.atleast_2d(np.asarray(X, dtype=float))
    T = np.atleast_2d(np.asarray(Y))
    if S.ndim != 2:
        raise ValueError(f"X must be 2-D of shape (m, K), got shape {S.shape}.")
    if T.shape != S.shape:
        raise ValueError(f"Y has shape {T.shape} but X has shape {S.shape}.")
    if not np.all(np.isfinite(S)):
        raise ValueError("X (scores) must be finite.")
    Ti = T.astype(int)
    if not np.all(np.isin(Ti, (0, 1))):
        raise ValueError("Y must contain only 0 and 1; multilabel targets are binary per label.")
    m, K = S.shape
    t = np.asarray(thresholds, dtype=float).ravel()
    if t.size == 1:
        t = np.full(K, float(t[0]))
    if t.size != K:
        raise ValueError(f"thresholds has {t.size} entries but there are {K} labels.")

    P = (S > t).astype(int)

    f1s, precs, recs = [], [], []
    for k in range(K):
        yk, pk = Ti[:, k], P[:, k]
        if not (np.any(yk == 1) or np.any(pk == 1)):
            f1s.append(0.0)
            precs.append(float("nan"))
            recs.append(float("nan"))
            continue
        rk = geron_f1_score(yk, pk, positive_class=1)
        f1s.append(rk["f1"])
        precs.append(rk["precision"])
        recs.append(rk["recall"])

    tp = int(np.sum((P == 1) & (Ti == 1)))
    fp = int(np.sum((P == 1) & (Ti == 0)))
    fn = int(np.sum((P == 0) & (Ti == 1)))
    micro = 0.0 if 2 * tp + fp + fn == 0 else 2.0 * tp / (2 * tp + fp + fn)

    return RichResult(
        title="Multilabel classification",
        summary_lines=[("Labels", int(K)), ("Macro F1", float(np.mean(f1s))),
                       ("Micro F1", micro)],
        payload={
            "predictions": P.tolist(),
            "per_label_f1": f1s,
            "per_label_precision": precs,
            "per_label_recall": recs,
            "macro_f1": float(np.mean(f1s)),
            "micro_f1": micro,
            "exact_match_ratio": float(np.mean(np.all(P == Ti, axis=1))),
            "hamming_loss": float(np.mean(P != Ti)),
            "thresholds": t.tolist(),
            "estimate": float(np.mean(f1s)),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmlb: y_hat_k = 1{score_k > t_k} per label; macro and micro F1 (per-label F1 via grf1)"
