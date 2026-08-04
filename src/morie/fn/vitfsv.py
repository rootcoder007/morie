# morie.fn -- slice s01 (rootcoder007/morie)
"""ViT fine-tuning: the downstream head attached to z_L^0.

SOURCE.  Dosovitskiy et al. (2021), "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale", ICLR 2021;
arXiv:2010.11929v2.  Read from the PDF rendered as page images.

Section 3.2 "Fine-tuning and higher resolution", p. 4: "Typically, we
pre-train ViT on large datasets, and fine-tune to (smaller) downstream
tasks.  For this, we remove the pre-trained prediction head and attach a
zero-initialized D x K feedforward layer, where K is the number of
downstream classes."  Section 3.1, p. 3, adds that the head is "a MLP
with one hidden layer at pre-training time and by a single linear layer
at fine-tuning time", attached to z_L^0.

Section 4.1 "Setup", Metrics, p. 5: "Few-shot accuracies are obtained by
solving a regularized least-squares regression problem that maps the
(frozen) representation of a subset of training images to {-1, 1}^K
target vectors.  This formulation allows us to recover the exact
solution in closed form."

Two modes are therefore implemented, and both are the paper's own:

  mode = "init"     the head exactly as Section 3.2 attaches it, all
                    zeros.  Every logit is 0, so with the first-maximum
                    tie rule every image is assigned class 1.  This is
                    the state of the model before any fine-tuning step
                    and is included because it is the one point where
                    the paper pins the head's numerical value.
  mode = "fewshot"  the closed-form regularized least-squares head of
                    Section 4.1, W = (X'X + lambda I)^{-1} X' T with
                    T the {-1, 1}^K target matrix and X the frozen
                    representations.

  mode = "full"     unfreezing the backbone requires backpropagation
                    through the encoder, which this package does not
                    implement.  It raises rather than pretending.

The paper does not state a tie-breaking rule for argmax; first maximum
is used, matching R's which.max, and is stated as this implementation's
choice.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core
from . import _vitcore as vc

from ._richresult import RichResult

__all__ = ["vit_finetune"]


def vit_finetune(model, data, mode="init", ridge=1e-8, n_classes=None):
    """Attach and evaluate the downstream classification head.

    Parameters
    ----------
    model : array-like
        n-by-D matrix of frozen representations; row i is the y of
        Equation (4) for image i.
    data : array-like
        Length-n integer class labels in 1 ... K.
    mode : str
        "init", "fewshot" or "full".
    ridge : float
        lambda of the regularized least squares, "fewshot" only.
    n_classes : int or None
        K.  Defaults to the largest label seen.

    Returns
    -------
    estimate  : accuracy
    head      : W, D-by-K
    logits    : n-by-K
    pred      : predicted class per row
    confusion : K-by-K, rows true class, columns predicted class
    """
    X = core.mat(model)
    n = len(X)
    if n < 1:
        raise ValueError("vit_finetune: model must have at least one row")
    d = len(X[0])
    for r in X:
        if len(r) != d:
            raise ValueError("vit_finetune: rows of model have unequal length")
    y = [int(t) for t in core.vec(data)]
    if len(y) != n:
        raise ValueError("vit_finetune: data must have one label per row of model")
    kk = max(y) if n_classes is None else int(n_classes)
    if kk < 1:
        raise ValueError("vit_finetune: n_classes must be at least 1")
    for t in y:
        if t < 1 or t > kk:
            raise ValueError("vit_finetune: labels must lie in 1 ... K")
    if mode == "full":
        raise ValueError(
            "vit_finetune: mode 'full' unfreezes the backbone, which needs "
            "backpropagation through the encoder; not implemented"
        )
    if mode not in ("init", "fewshot"):
        raise ValueError("vit_finetune: mode must be 'init', 'fewshot' or 'full'")
    if mode == "init":
        W = [[0.0] * kk for _ in range(d)]
    else:
        T = [[1.0 if y[i] == c + 1 else -1.0 for c in range(kk)] for i in range(n)]
        W = [[0.0] * kk for _ in range(d)]
        for c in range(kk):
            col = core.lstsq(X, [T[i][c] for i in range(n)], float(ridge))
            for j in range(d):
                W[j][c] = col[j]
    logits = core.matmul(X, W)
    pred = [vc.argmax_first(r) + 1 for r in logits]
    conf = [[0] * kk for _ in range(kk)]
    hit = 0
    for i in range(n):
        conf[y[i] - 1][pred[i] - 1] += 1
        if pred[i] == y[i]:
            hit += 1
    return RichResult(
        title="ViT fine-tuning head",
        summary_lines=[("images", n), ("embed dim", d), ("classes", kk), ("mode", mode)],
        payload={
            "estimate": hit / n,
            "accuracy": hit / n,
            "head": W,
            "logits": logits,
            "pred": pred,
            "confusion": conf,
            "n_correct": hit,
            "n_classes": kk,
            "embed_dim": d,
            "mode": mode,
            "n": n,
            "method": "zero-initialised D x K head (Dosovitskiy et al. 2021, Sec. 3.2 p. 4); closed-form regularized least squares onto {-1,1}^K (Sec. 4.1 Metrics p. 5)",
        },
    )


def cheatsheet():
    return "vitfsv: ViT downstream head -- zero-init D x K layer, or closed-form few-shot least squares"


# compact alias per ledger/NAMING.md
vitfinetune = vit_finetune
