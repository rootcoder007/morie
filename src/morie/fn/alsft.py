# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SetFit two-step: contrastive pairs then a logistic head
(Tunstall et al. 2022; Alammar Ch 11)."""

from . import _array_core as np

from ._richresult import RichResult
from .alembc import alammar_embedding_classifier

__all__ = ["alammar_setfit_twostep"]


def alammar_setfit_twostep(embeddings, labels, n_pairs_report=True):
    """Step 1 builds the contrastive pair set (same-class positive,
    cross-class negative -- the pair GENERATION is the SetFit recipe);
    step 2 fits the classification head on the embeddings, natively.
    The encoder fine-tune itself belongs to the caller's model; what
    is computed here is everything the paper specifies around it.

    References: Alammar and Grootendorst, Ch 11; Tunstall et al.
    (2022).
    """
    X = np.atleast_2d(np.asarray(embeddings, dtype=float))
    y = np.atleast_1d(np.asarray(labels)).astype(int)
    if X.shape[0] != len(y):
        raise ValueError("need one label per embedding.")
    n = len(y)
    pos = []
    neg = []
    for i in range(n):
        for j in range(i + 1, n):
            (pos if y[i] == y[j] else neg).append((i, j))
    if not pos or not neg:
        raise ValueError(
            "contrastive pairs need at least two classes with at least "
            "two members each.")
    head = alammar_embedding_classifier(X, y)
    return RichResult(payload={
        "positive_pairs": pos, "negative_pairs": neg,
        "n_positive": len(pos), "n_negative": len(neg),
        "head_train_accuracy": head["train_accuracy"],
        "head_predictions": head["predictions"],
        "estimate": head["train_accuracy"], "n": n,
        "method": "SetFit pair generation + head (Tunstall et al. 2022)"})


def cheatsheet():
    return "alsft: same/cross-class pair sets + native logistic head"
