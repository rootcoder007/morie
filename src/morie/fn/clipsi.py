# morie.fn -- function file (rootcoder007/morie)
"""CLIP image-text cosine similarity, temperature scaled."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["clip_similarity"]


def l2_normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    if n <= 0.0:
        raise ValueError("cannot normalise a zero-norm embedding")
    return [x / n for x in v]


def clip_similarity(I_emb, T_emb, tau=0.01):
    """
    CLIP image-text similarity

    Formula: cos(I_emb, T_emb) / tau

    Both embeddings are L2-normalised first, so the inner product IS the
    cosine and the logit is bounded by +/- 1/tau.  The diagonal of the
    logit matrix holds the matched pairs; retrieval takes the row
    argmax, and the reported accuracy is how often that argmax is the
    diagonal.

    Parameters
    ----------
    I_emb : array-like
        n x d matrix of image embeddings.
    T_emb : array-like
        n x d matrix of text embeddings, paired row by row.
    tau : float
        Temperature, strictly positive.

    Returns
    -------
    result : dict
        Keys: estimate (mean matched cosine), logits, cosine,
        retrieved, accuracy, n, d.

    References
    ----------
    Radford et al. (2021), Learning Transferable Visual Models From
    Natural Language Supervision, ICML 139:8748-8763.
    """
    I = core.mat(I_emb)
    T = core.mat(T_emb)
    n = len(I)
    if n == 0:
        raise ValueError("empty input: I_emb has no rows")
    if len(T) != n:
        raise ValueError("I_emb and T_emb must have the same number of rows")
    d = len(I[0])
    if len(T[0]) != d:
        raise ValueError("image and text embeddings must share a dimension")
    if not (tau > 0.0):
        raise ValueError("tau must be strictly positive")
    In = [l2_normalize(r) for r in I]
    Tn = [l2_normalize(r) for r in T]
    cos = [[sum(In[i][k] * Tn[j][k] for k in range(d)) for j in range(n)]
           for i in range(n)]
    logits = [[cos[i][j] / tau for j in range(n)] for i in range(n)]
    retrieved = []
    for i in range(n):
        b = 0
        for j in range(n):
            if cos[i][j] > cos[i][b]:
                b = j
        retrieved.append(b)
    acc = sum(1 for i in range(n) if retrieved[i] == i) / float(n)
    return RichResult(payload={
        "estimate": sum(cos[i][i] for i in range(n)) / n,
        "logits": logits,
        "cosine": cos,
        "retrieved": retrieved,
        "accuracy": acc,
        "n": n,
        "d": d,
        "method": "CLIP image-text cosine similarity",
    })


def cheatsheet():
    return "clipsi: CLIP image-text cosine similarity"


# compact alias per ledger/NAMING.md
clipsimilarity = clip_similarity
