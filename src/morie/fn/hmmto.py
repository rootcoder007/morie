# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multioutput classification."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_multioutput"]


def geron_multioutput(X, Y, k=1, X_new=None):
    """
    Multioutput classification: predict several categorical targets per instance.

    Formula: Y_hat = f(X); Y in discrete grid per output

    Multilabel with more than two values per label. Geron's example is
    denoising: every PIXEL is one output whose classes are the intensity
    levels, so a single model outputs a whole image. The k-nearest
    neighbour vote used here is exactly the model he uses there, and it
    generalises to any number of outputs at no extra cost, because the
    neighbours are found ONCE and each output votes among them
    separately.

    With no ``X_new`` the predictions are leave-one-out (each row's own
    neighbourhood excludes itself), so the reported accuracy is honest
    rather than the trivially perfect 1-NN training score.

    Parameters
    ----------
    X : array-like, shape (m, n)
    Y : array-like, shape (m, t)
        Categorical targets, one column per output.
    k : int, default 1
        Neighbours voting.
    X_new : array-like, optional
        Rows to predict; default leave-one-out over ``X``.

    Returns
    -------
    result : RichResult
        Keys: predictions, predict, accuracy_per_output, accuracy,
        n_outputs, classes_per_output, estimate, n, method.

    Examples
    --------
    Two clusters, two outputs; leave-one-out 1-NN gets both right:

    >>> X = [[0.0], [1.0], [10.0], [11.0]]
    >>> Y = [[0, 1], [0, 1], [1, 0], [1, 0]]
    >>> r = geron_multioutput(X, Y)
    >>> r["predictions"].tolist()
    [[0, 1], [0, 1], [1, 0], [1, 0]]
    >>> float(r["accuracy"]), [float(a) for a in r["accuracy_per_output"]]
    (1.0, [1.0, 1.0])

    Predicting a new point uses all the training rows:

    >>> r["predict"]([[10.5]]).tolist()
    [[1, 0]]

    References
    ----------
    Geron Ch 3
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_multioutput: X must be a non-empty 2-D array, got shape {A.shape}")
    Yv = np.asarray(Y)
    if Yv.ndim == 1:
        Yv = Yv.reshape(-1, 1)
    if Yv.ndim != 2:
        raise ValueError(f"geron_multioutput: Y must be 2-D (one column per output), got ndim={Yv.ndim}")
    m, t = Yv.shape
    if m != A.shape[0]:
        raise ValueError(f"geron_multioutput: X has {A.shape[0]} rows but Y has {m}")
    kk = int(k)
    if kk < 1:
        raise ValueError(f"geron_multioutput: k must be >= 1, got {k!r}")
    if kk > m - 1 and X_new is None:
        raise ValueError(f"geron_multioutput: k={kk} needs {kk + 1} rows for leave-one-out, only {m} given")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_multioutput: X contains non-finite values")

    def _vote(D, exclude_self):
        out = np.empty((D.shape[0], t), dtype=Yv.dtype)
        for i in range(D.shape[0]):
            order = np.argsort(D[i], kind="mergesort")
            if exclude_self:
                order = order[order != i]
            nb = order[:kk]
            for j in range(t):
                vals, cnt = np.unique(Yv[nb, j], return_counts=True)
                out[i, j] = vals[np.argmax(cnt)]
        return out

    def predict(Xnew, _A=A, _d=A.shape[1]):
        B = np.atleast_2d(np.asarray(Xnew, dtype=float))
        if B.shape[1] != _d:
            raise ValueError(f"predict: expected {_d} features, got {B.shape[1]}")
        D = np.sqrt(((B[:, None, :] - _A[None, :, :]) ** 2).sum(axis=2))
        return _vote(D, exclude_self=False)

    if X_new is None:
        D = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(axis=2))
        pred = _vote(D, exclude_self=True)
        per = [float(np.mean(pred[:, j] == Yv[:, j])) for j in range(t)]
        acc = float(np.mean(pred == Yv))
        exact = float(np.mean(np.all(pred == Yv, axis=1)))
    else:
        pred = predict(X_new)
        per = [float("nan")] * t
        acc = float("nan")
        exact = float("nan")

    return RichResult(
        title="Multioutput classification",
        summary_lines=[("Outputs", int(t)), ("Neighbours", kk), ("Mean accuracy", acc)],
        interpretation="One neighbourhood serves every output, so extra outputs cost almost nothing.",
        payload={
            "predictions": pred,
            "predict": predict,
            "accuracy_per_output": per,
            "accuracy": acc,
            "exact_match": exact,
            "n_outputs": int(t),
            "classes_per_output": [np.unique(Yv[:, j]) for j in range(t)],
            "estimate": pred,
            "n": int(m),
            "method": f"{kk}-NN multioutput vote (leave-one-out when no new rows are given)",
        },
    )


def cheatsheet():
    return "hmmto: Multioutput classification by k-NN voting per output"
