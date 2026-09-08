# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multilabel classification: predict a subset of labels per instance."""

from . import _array_core as np

from ._richresult import RichResult
from .grcfm import geron_confusion_matrix

__all__ = ["geron_multilabel"]

_METHOD = "Multilabel classification (binary relevance kNN, leave-one-out)"


def geron_multilabel(X, Y, k=3, Y_pred=None):
    """
    Multilabel classification: predict a subset of labels per instance.

    Formula: y_i in {0,1}^K; Hamming / Jaccard loss

    Each instance carries a *subset* of labels, so accuracy has to be
    defined before it can be reported, and the three usual definitions
    disagree badly:

    ``subset_accuracy`` -- fraction of instances whose predicted set is
    exactly right.  Brutal: one wrong label out of twenty scores zero.

    ``hamming_loss`` -- fraction of individual label decisions that are
    wrong.  Forgiving, and misleading when labels are sparse: predicting
    all-zeros scores well when 5% of labels are positive.  The
    all-zeros baseline is returned for exactly that comparison.

    ``jaccard`` -- mean ``|intersection| / |union|`` per instance, which
    is the compromise most multilabel work reports.

    Predictions come either from ``Y_pred`` or, by default, from a
    leave-one-out binary-relevance kNN: each label is predicted
    independently by majority vote among the ``k`` nearest *other*
    instances.  Binary relevance ignores label correlations by
    construction -- that is its known weakness, and the measured
    per-label F1 (delegated to
    :func:`morie.fn.grcfm.geron_confusion_matrix`) is where it shows.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Features.
    Y : array-like, shape (m, K)
        Binary label matrix.
    k : int
        Neighbours for the default predictor.
    Y_pred : array-like, shape (m, K), optional
        Supply predictions instead of fitting.

    Returns
    -------
    result : RichResult
        Keys: Y_pred, subset_accuracy, hamming_loss, jaccard,
        per_label_f1, macro_f1, zero_baseline_hamming, estimate, n, method.

    Examples
    --------
    Hand-checkable metrics from supplied predictions.  Two instances,
    three labels, two wrong decisions out of six:

    >>> Y = [[1, 0, 1], [0, 1, 1]]
    >>> P = [[1, 0, 0], [0, 1, 1]]
    >>> r = geron_multilabel([[0.0], [1.0]], Y, Y_pred=P)
    >>> round(r["hamming_loss"], 6)
    0.166667
    >>> float(r["subset_accuracy"])
    0.5

    Jaccard per instance: the first is ``1/2`` (one of two shared out of
    two in the union), the second is 1, so the mean is 0.75:

    >>> float(r["jaccard"])
    0.75

    Sparse labels make Hamming loss flattering -- the all-zeros baseline
    is reported so that is visible:

    >>> sparse = [[0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0]]
    >>> s = geron_multilabel([[0.0], [1.0], [2.0]], sparse,
    ...                      Y_pred=[[0, 0, 0, 0]] * 3)
    >>> round(s["hamming_loss"], 6) == round(s["zero_baseline_hamming"], 6)
    True
    >>> float(s["subset_accuracy"])
    0.3333333333333333

    The default kNN predictor recovers well-separated label patterns:

    >>> Xk = [[0.0], [0.1], [0.2], [9.0], [9.1], [9.2]]
    >>> Yk = [[1, 0]] * 3 + [[0, 1]] * 3
    >>> g = geron_multilabel(Xk, Yk, k=2)
    >>> float(g["subset_accuracy"])
    1.0

    References
    ----------
    Géron Ch 3
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_multilabel: X must be a non-empty 2-D array, got shape {A.shape}")
    YY = np.atleast_2d(np.asarray(Y))
    if YY.ndim != 2 or YY.size == 0:
        raise ValueError(f"geron_multilabel: Y must be a non-empty (m, K) label matrix, got shape {YY.shape}")
    if YY.shape[0] != A.shape[0]:
        raise ValueError(f"geron_multilabel: X has {A.shape[0]} rows but Y has {YY.shape[0]}")
    Yb = np.asarray(YY, dtype=float)
    if not np.all(np.isin(Yb, (0.0, 1.0))):
        raise ValueError(
            f"geron_multilabel: Y must be binary; got distinct values {np.unique(Yb).tolist()}"
        )
    Yb = Yb.astype(int)
    m, K = Yb.shape

    if Y_pred is not None:
        P = np.atleast_2d(np.asarray(Y_pred, dtype=float))
        if P.shape != Yb.shape:
            raise ValueError(f"geron_multilabel: Y_pred has shape {P.shape} but Y has {Yb.shape}")
        if not np.all(np.isin(P, (0.0, 1.0))):
            raise ValueError("geron_multilabel: Y_pred must be binary")
        P = P.astype(int)
        source = "supplied"
    else:
        kk = int(k)
        if not (1 <= kk <= m - 1):
            raise ValueError(
                f"geron_multilabel: k must lie in 1..{m - 1} for leave-one-out prediction, got {k!r}"
            )
        D = np.sqrt(np.clip(np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=2), 0.0, None))
        np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, axis=1, kind="mergesort")[:, :kk]
        P = (Yb[nn].mean(axis=1) >= 0.5).astype(int)
        source = f"leave-one-out {kk}-NN binary relevance"

    correct_cells = P == Yb
    hamming = float(1.0 - np.mean(correct_cells))
    subset = float(np.mean(np.all(correct_cells, axis=1)))

    inter = np.sum((P == 1) & (Yb == 1), axis=1)
    union = np.sum((P == 1) | (Yb == 1), axis=1)
    jac = np.where(union > 0, inter / np.where(union == 0, 1, union), 1.0)
    jaccard = float(np.mean(jac))

    f1s = []
    for j in range(K):
        cm = geron_confusion_matrix(Yb[:, j], P[:, j], n_classes=2)
        f1s.append(float(cm["f1"][1]))
    macro = float(np.mean(f1s))

    zero_baseline = float(np.mean(Yb))

    warns = []
    if zero_baseline < 0.2:
        warns.append(
            f"only {zero_baseline:.1%} of label cells are positive: an all-zeros predictor would score "
            f"a Hamming loss of {zero_baseline:.4g}. Read subset accuracy or Jaccard instead."
        )

    return RichResult(
        title="Multilabel classification",
        summary_lines=[
            ("Instances x labels", f"{m} x {K}"),
            ("Subset accuracy", subset),
            ("Hamming loss", hamming),
            ("Jaccard", jaccard),
            ("Macro F1", macro),
        ],
        warnings=warns,
        interpretation=(
            "Subset accuracy is the strictest reading and Hamming the most forgiving; on sparse "
            "labels only the strict ones distinguish a model from an all-zeros predictor."
        ),
        payload={
            "Y_pred": P,
            "subset_accuracy": subset,
            "hamming_loss": hamming,
            "jaccard": jaccard,
            "per_label_f1": f1s,
            "macro_f1": macro,
            "zero_baseline_hamming": zero_baseline,
            "source": source,
            "estimate": subset,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmlb: multilabel metrics -- subset accuracy, Hamming, Jaccard, per-label F1 (via grcfm)"
