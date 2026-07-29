# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Softmax classifier on frozen embeddings (Alammar Ch 4)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_embedding_classifier"]


def alammar_embedding_classifier(embeddings, labels, n_steps=500,
                                 learning_rate=0.5, l2=1e-4):
    """p(y | x) = softmax(W emb(x) + b), trained here by full-batch
    gradient descent on the cross-entropy -- a REAL trainable head,
    not a stub. The encoder stays frozen by construction: this
    function only ever sees its outputs.

    References: Alammar and Grootendorst, Ch 4.
    """
    X = np.atleast_2d(np.asarray(embeddings, dtype=float))
    y = np.atleast_1d(np.asarray(labels)).astype(int)
    if X.shape[0] != len(y):
        raise ValueError("need one label per embedding.")
    classes = sorted(set(int(v) for v in y))
    if classes != list(range(len(classes))):
        raise ValueError("labels must be 0..K-1 with every class present.")
    K = len(classes)
    if K < 2:
        raise ValueError("need at least 2 classes.")
    n, d = X.shape
    W = np.zeros((K, d)); b = np.zeros(K)
    lr = float(learning_rate)
    for _ in range(int(n_steps)):
        Zl = X @ W.T + b
        Zl -= Zl.max(axis=1, keepdims=True)
        P = np.exp(Zl) / np.exp(Zl).sum(axis=1, keepdims=True)
        G = P.copy()
        G[np.arange(n), y] -= 1.0
        W -= lr * (G.T @ X / n + float(l2) * W)
        b -= lr * G.mean(axis=0)
    Zl = X @ W.T + b
    Zl -= Zl.max(axis=1, keepdims=True)
    P = np.exp(Zl) / np.exp(Zl).sum(axis=1, keepdims=True)
    pred = np.argmax(P, axis=1)
    acc = float(np.mean(pred == y))
    ce = float(np.mean(-np.log(P[np.arange(n), y] + 1e-12)))
    return RichResult(payload={
        "weights": [[float(v) for v in r] for r in W],
        "bias": [float(v) for v in b],
        "train_accuracy": acc, "cross_entropy": ce,
        "predictions": [int(v) for v in pred],
        "estimate": acc, "n": n,
        "method": "Softmax head on frozen embeddings (Alammar Ch 4)"})


def cheatsheet():
    return "alembc: trainable softmax head; encoder frozen by construction"
