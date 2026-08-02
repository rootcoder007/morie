# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Supervised fine-tuning (SFT) on instruction-response pairs."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_sft"]


def _featurise(data):
    """Bag-of-words prompt features and response-token targets.

    Accepts ``(prompt, response)`` pairs where the prompt is a string or a
    numeric vector and the response is a string token or an integer id.
    """
    prompts, responses = [], []
    for i, item in enumerate(data):
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            raise ValueError(f"geron_sft: item {i} is not an (instruction, response) pair")
        prompts.append(item[0])
        responses.append(item[1])
    if isinstance(prompts[0], str):
        vocab = sorted({w for p in prompts for w in str(p).lower().split()})
        if not vocab:
            raise ValueError("geron_sft: the instructions contain no tokens")
        idx = {w: j for j, w in enumerate(vocab)}
        X = np.zeros((len(prompts), len(vocab)))
        for i, p in enumerate(prompts):
            for w in str(p).lower().split():
                X[i, idx[w]] += 1.0
    else:
        X = np.asarray([np.asarray(p, dtype=float).ravel() for p in prompts], dtype=float)
        vocab = None
        idx = None
    if isinstance(responses[0], str):
        labels = sorted(set(responses))
        lidx = {t: j for j, t in enumerate(labels)}
        y = np.asarray([lidx[r] for r in responses], dtype=int)
    else:
        y = np.asarray(responses, dtype=int)
        labels = sorted(set(y.tolist()))
        lidx = {t: j for j, t in enumerate(labels)}
        y = np.asarray([lidx[int(r)] for r in y], dtype=int)
    return X, y, vocab, idx, labels


def geron_sft(model=None, instruction_data=None, epochs=200, lr=0.5, l2=0.0):
    """
    Supervised fine-tuning (SFT) on instruction-response pairs.

    Formula: L = -sum_i log P(y_i | x_i) over instruction dataset

    SFT is ordinary maximum-likelihood training on demonstrations, and
    that is what runs here: the instruction is featurised (bag of words
    for text prompts, or used as-is for numeric ones), the response is the
    target token, and the softmax head is trained by gradient descent on
    the exact cross-entropy gradient ``X^T (P - Y) / n``. The *sum* form
    of the loss from the formula is reported alongside the mean, because
    they differ by the dataset size and comparisons across datasets need
    the mean.

    Parameters
    ----------
    model : array-like, optional
        Initial weights (n_features, n_responses); default zeros -- the
        "before fine-tuning" state whose loss is reported for comparison.
    instruction_data : sequence of (instruction, response)
        Demonstrations. Required, non-empty, at least 2 distinct responses.
    epochs : int, default 200
        Gradient steps (>= 1).
    lr : float, default 0.5
        Learning rate (> 0).
    l2 : float, default 0.0
        Weight decay (>= 0).

    Returns
    -------
    result : RichResult
        Keys: W, loss, sum_loss, loss_curve, accuracy, predicted, vocab,
        labels, estimate, n, method.

    Examples
    --------
    Two instructions with disjoint vocabulary and different responses:
    before fine-tuning the model is uniform (loss log 2 per example), and
    training drives the loss down and the accuracy to 1.

    >>> import numpy as np
    >>> data = [("translate hello", "bonjour"), ("summarise text", "resume")]
    >>> r = geron_sft(None, data, epochs=300, lr=0.5)
    >>> round(float(r["loss_curve"][0]), 9) == round(float(np.log(2)), 9)
    True
    >>> bool(r["loss"] < 0.05)
    True
    >>> float(r["accuracy"])
    1.0
    >>> round(float(r["sum_loss"] / r["loss"]), 9)
    2.0

    References
    ----------
    Géron Ch 15
    """
    if instruction_data is None:
        raise ValueError("geron_sft: instruction_data is required -- SFT trains on demonstrations")
    data = list(instruction_data)
    if not data:
        raise ValueError("geron_sft: instruction_data is empty")
    X, y, vocab, idx, labels = _featurise(data)
    n, d = X.shape
    K = len(labels)
    if K < 2:
        raise ValueError(f"geron_sft: need at least 2 distinct responses to train a softmax head, got {K}")
    if not np.all(np.isfinite(X)):
        raise ValueError("geron_sft: the featurised instructions contain non-finite values")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_sft: epochs must be >= 1, got {E}")
    step = float(lr)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"geron_sft: lr must be positive and finite, got {step}")
    decay = float(l2)
    if not np.isfinite(decay) or decay < 0:
        raise ValueError(f"geron_sft: l2 must be non-negative and finite, got {decay}")

    W = np.zeros((d, K)) if model is None else np.asarray(model, dtype=float)
    if W.shape != (d, K):
        raise ValueError(f"geron_sft: model must have shape {(d, K)}, got {W.shape}")
    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1.0

    def _fwd(W):
        z = X @ W
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        P = e / e.sum(axis=1, keepdims=True)
        ll = float(-np.mean(np.log(np.maximum(P[np.arange(n), y], np.finfo(float).tiny))))
        return P, ll

    losses = []
    for _ in range(E):
        P, ll = _fwd(W)
        losses.append(ll)
        W = W - step * (X.T @ (P - Y) / n + decay * W)
    P, ll = _fwd(W)
    losses.append(ll)
    pred = np.argmax(P, axis=1)

    return RichResult(
        title="Supervised fine-tuning",
        summary_lines=[
            ("Demonstrations", n),
            ("Response classes", K),
            ("Mean NLL", ll),
            ("Accuracy", float(np.mean(pred == y))),
        ],
        interpretation=(
            "SFT only teaches the model to imitate the demonstrations it was given; it cannot express "
            "a preference between two plausible answers, which is what preference tuning is for."
        ),
        payload={
            "W": W,
            "loss": ll,
            "sum_loss": float(ll * n),
            "loss_curve": np.asarray(losses, dtype=float),
            "accuracy": float(np.mean(pred == y)),
            "predicted": np.asarray([labels[i] for i in pred], dtype=object),
            "probabilities": P,
            "vocab": vocab,
            "labels": labels,
            "estimate": ll,
            "n": int(n),
            "method": "Softmax maximum-likelihood fine-tuning on instruction-response demonstrations",
        },
    )


def cheatsheet():
    return "hmsft: Supervised fine-tuning (SFT) on instruction-response pairs"
