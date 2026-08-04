"""ViT fine-tuning head for a downstream task (Section 3.2)."""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["vit_finetune"]


def vit_finetune(model, data, mode="linear", steps=200, lr=0.5, eps=1e-5):
    """Attach a zero-initialised D-by-K head to frozen representations.

    Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words*, ICLR
    2021, arXiv:2010.11929v2, Section 3.2, p. 4: "we remove the
    pre-trained prediction head and attach a zero-initialized D x K
    feedforward layer, where K is the number of downstream classes",
    and Section 3.1, p. 3: the head sits on z_L^0, the representation y
    of Eq. (4).  Section B.1.1, p. 13, fine-tunes with SGD and momentum
    0.9; this arm uses plain full-batch gradient descent instead, with
    a fixed step count, because both language arms must land on the
    same numbers and momentum adds nothing that a determinism check
    would notice.  That substitution is an implementation choice, not
    the paper's recipe.

    ``mode`` is the frozen/unfrozen distinction of Section 3.2 reduced
    to what this API actually holds -- representations, not a network:
    "linear" trains the head alone; "full" also trains a per-feature
    affine recalibration of the representation, standing in for
    unfreezing the backbone.  It cannot be back-propagation into a
    backbone this function was never given.

    Parameters
    ----------
    model : array-like
        n-by-D matrix of representations, one row per image (the y of
        Eq. (4), e.g. from ``vit_forward``).
    data : array-like
        Length-n integer class labels in 1 ... K.
    mode : str
        "linear" or "full".
    steps : int
        Gradient-descent steps.
    lr : float
        Step size.
    eps : float
        Guard inside the log of the cross-entropy.

    Returns
    -------
    estimate    : training accuracy of the fine-tuned head
    predictions : length-n predicted labels (argmax of the logits,
                  ties going to the lowest label)
    loss        : final mean cross-entropy; at step 0 it is log K
    W, b        : the K columns of the head and its bias
    """
    X = core.mat(model)
    n = len(X)
    if n == 0:
        raise ValueError("vit_finetune: model is empty")
    D = len(X[0])
    for r in X:
        if len(r) != D:
            raise ValueError("vit_finetune: model is ragged")
    y = [int(e) for e in core.vec(data)]
    if len(y) != n:
        raise ValueError("vit_finetune: data must have one label per row of model")
    K = max(y)
    if min(y) < 1:
        raise ValueError("vit_finetune: labels must be integers 1 ... K")
    if K < 2:
        raise ValueError("vit_finetune: need at least two classes")
    if mode not in ("linear", "full"):
        raise ValueError("vit_finetune: mode must be 'linear' or 'full'")
    ns = int(steps)
    if ns < 0:
        raise ValueError("vit_finetune: steps must not be negative")
    W = [[0.0] * K for _ in range(D)]
    b = [0.0] * K
    a = [1.0] * D
    c = [0.0] * D
    full = mode == "full"
    loss = 0.0
    for _ in range(ns + 1):
        Z = [[a[j] * X[i][j] + c[j] for j in range(D)] for i in range(n)]
        P = []
        loss = 0.0
        for i in range(n):
            lg = [b[t] for t in range(K)]
            for j in range(D):
                zij = Z[i][j]
                for t in range(K):
                    lg[t] += zij * W[j][t]
            p = core.softmax(lg)
            P.append(p)
            loss -= math.log(p[y[i] - 1] + eps)
        loss /= n
        if _ == ns:
            break
        gW = [[0.0] * K for _ in range(D)]
        gb = [0.0] * K
        ga = [0.0] * D
        gc = [0.0] * D
        for i in range(n):
            for t in range(K):
                d = P[i][t] - (1.0 if y[i] == t + 1 else 0.0)
                gb[t] += d / n
                for j in range(D):
                    gW[j][t] += Z[i][j] * d / n
                    if full:
                        ga[j] += d * W[j][t] * X[i][j] / n
                        gc[j] += d * W[j][t] / n
        for j in range(D):
            for t in range(K):
                W[j][t] -= lr * gW[j][t]
        for t in range(K):
            b[t] -= lr * gb[t]
        if full:
            for j in range(D):
                a[j] -= lr * ga[j]
                c[j] -= lr * gc[j]
    pred = []
    hit = 0
    for i in range(n):
        best = 0
        for t in range(1, K):
            if P[i][t] > P[i][best]:
                best = t
        pred.append(best + 1)
        if best + 1 == y[i]:
            hit += 1
    return RichResult(payload={
        "estimate": hit / n,
        "predictions": pred,
        "probs": P,
        "loss": loss,
        "W": W,
        "b": b,
        "gain": a,
        "shift": c,
        "n": n,
        "n_classes": K,
        "embed_dim": D,
        "mode": mode,
        "method": "ViT fine-tune for downstream",
    })


def cheatsheet():
    return "vitfsv: ViT fine-tune for downstream"
