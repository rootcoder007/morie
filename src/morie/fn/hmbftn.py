# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fine-tune BERT on a downstream classification task."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_bert_finetune"]


def _softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def geron_bert_finetune(bert, X, y, epochs=100, lr=0.1, l2=0.0):
    """
    Fine-tune BERT on a downstream classification task.

    Formula: classifier head on [CLS]; train end-to-end on task data

    `bert` is any callable ``bert(X) -> (n, d)`` pooled [CLS] embeddings --
    a frozen encoder, the output of :func:`morie.fn.hmbert.geron_bert`, or a
    stub. A softmax classification head is then trained on top by full-batch
    gradient descent on the cross-entropy, with the exact gradient
    ``X^T (p - onehot) / n``.

    Parameters
    ----------
    bert : callable
        Embedding function; must return one row per input example.
    X : array-like
        Inputs passed straight to `bert`.
    y : array-like of int
        Class labels 0..K-1.
    epochs : int
        Gradient steps (>= 1).
    lr : float
        Step size (positive).
    l2 : float
        Optional L2 penalty on the head weights (non-negative); the bias is
        left unpenalised.

    Returns
    -------
    result : RichResult
        Keys: W, b, losses, accuracy, predict, embeddings, estimate, n, method.

    Examples
    --------
    Identity "encoder" on two orthogonal, perfectly separable examples:

    >>> ident = lambda A: np.asarray(A, dtype=float)
    >>> r = geron_bert_finetune(ident, [[1.0, 0.0], [0.0, 1.0]], [0, 1], epochs=200, lr=0.5)
    >>> float(r["accuracy"])
    1.0
    >>> bool(r["losses"][-1] < r["losses"][0])
    True
    >>> bool(np.all(np.diff(r["losses"]) <= 1e-12))
    True

    The starting loss of a zero-initialised head is log K exactly:

    >>> round(float(r["losses"][0]), 6)
    0.693147
    >>> [int(v) for v in r["predict"]([[2.0, 0.0], [0.0, 3.0]])]
    [0, 1]

    References
    ----------
    Géron Ch 15
    """
    if not callable(bert):
        raise ValueError("geron_bert_finetune: bert must be callable, returning pooled [CLS] embeddings")
    labels = np.asarray(y).ravel()
    if labels.size == 0:
        raise ValueError("geron_bert_finetune: y is empty")
    if not np.all(labels == np.floor(np.asarray(labels, dtype=float))):
        raise ValueError("geron_bert_finetune: y must contain integer class labels")
    labels = labels.astype(int)
    if labels.min() < 0:
        raise ValueError("geron_bert_finetune: class labels must be non-negative")

    Z = np.asarray(bert(X), dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.ndim != 2:
        raise ValueError(f"geron_bert_finetune: bert must return a 2-D (n, d) array, got ndim={Z.ndim}")
    n, dim = Z.shape
    if n != labels.size:
        raise ValueError(f"geron_bert_finetune: bert returned {n} embeddings but y has {labels.size} labels")
    if not np.all(np.isfinite(Z)):
        raise ValueError("geron_bert_finetune: bert returned non-finite embeddings")
    K = int(labels.max()) + 1
    if K < 2:
        raise ValueError("geron_bert_finetune: need at least 2 classes to fine-tune a classifier head")
    EP = int(epochs)
    if EP < 1:
        raise ValueError("geron_bert_finetune: epochs must be >= 1")
    step = float(lr)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("geron_bert_finetune: lr must be a positive finite step size")
    lam = float(l2)
    if lam < 0:
        raise ValueError("geron_bert_finetune: l2 must be non-negative")

    W = np.zeros((dim, K))
    b = np.zeros(K)
    Y = np.zeros((n, K))
    Y[np.arange(n), labels] = 1.0
    losses = np.empty(EP)
    for e in range(EP):
        p = _softmax(Z @ W + b)
        losses[e] = float(-np.mean(np.log(np.clip(p[np.arange(n), labels], 1e-15, None))) + 0.5 * lam * np.sum(W * W))
        G = (p - Y) / n
        W = W - step * (Z.T @ G + lam * W)
        b = b - step * G.sum(axis=0)

    p_final = _softmax(Z @ W + b)
    pred = np.argmax(p_final, axis=1)
    acc = float(np.mean(pred == labels))

    def predict(Xnew, _bert=bert, _W=W, _b=b, _d=dim):
        Zn = np.asarray(_bert(Xnew), dtype=float)
        if Zn.ndim == 1:
            Zn = Zn.reshape(1, -1)
        if Zn.shape[1] != _d:
            raise ValueError(f"predict: encoder returned width {Zn.shape[1]}, expected {_d}")
        return np.argmax(Zn @ _W + _b, axis=1)

    return RichResult(
        title="BERT fine-tuning (classification head)",
        summary_lines=[("Classes", K), ("Final loss", float(losses[-1])), ("Training accuracy", acc)],
        payload={
            "W": W,
            "b": b,
            "losses": losses,
            "accuracy": acc,
            "predict": predict,
            "probabilities": p_final,
            "embeddings": Z,
            "estimate": float(losses[-1]),
            "n": int(n),
            "method": "Softmax classification head fine-tuned on pooled [CLS] embeddings",
        },
    )


def cheatsheet():
    return "hmbftn: Fine-tune BERT on a downstream classification task"
