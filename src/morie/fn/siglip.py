# morie.fn -- function file (rootcoder007/morie)
"""SigLIP pairwise sigmoid loss."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["siglip_pairwise"]


def siglip_pairwise(image_emb, text_emb, t_prime=1.0, bias=0.0):
    """Contrastive loss that scores each pair on its own.

    CLIP softmax normalises over the whole batch, so every pair's loss
    depends on every other pair and the batch has to be huge and
    globally synchronised.  Replacing it with a sigmoid per pair removes
    that coupling entirely: the loss decomposes, it works at small batch
    size, and it needs no all-gather.  The learnable bias exists because
    a batch is overwhelmingly negatives, and without it the sigmoid
    starts badly miscalibrated.

    Formula: ``L = -(1/|B|) sum_i sum_j log sigmoid(z_ij (t x_i . y_j + b))``
    with ``z_ij = 1`` on the diagonal and ``-1`` elsewhere.

    Parameters
    ----------
    image_emb : array-like, shape (n, d)
        Image embeddings; L2-normalised here.
    text_emb : array-like, shape (n, d)
        Text embeddings; L2-normalised here.
    t_prime : float, default 1.0
        Logit scale.
    bias : float, default 0.0
        Logit bias.

    Returns
    -------
    RichResult
        ``estimate`` (the loss), ``logits``, ``acc`` (fraction of rows
        whose diagonal is the largest logit), ``n``.

    References
    ----------
    Zhai, X., Mustafa, B., Kolesnikov, A. & Beyer, L. (2023).  Sigmoid
    loss for language image pre-training.  ICCV 2023, equation (3).
    """
    A = C.mat(image_emb)
    B = C.mat(text_emb)
    n = len(A)
    d = len(A[0])
    def unit(M):
        out = []
        for row in M:
            nm = math.sqrt(sum(v * v for v in row))
            out.append([v / nm if nm > 0 else 0.0 for v in row])
        return out
    A, B = unit(A), unit(B)
    logits = [[t_prime * sum(A[i][k] * B[j][k] for k in range(d)) + bias
               for j in range(n)] for i in range(n)]
    loss = 0.0
    hit = 0
    for i in range(n):
        for j in range(n):
            z = 1.0 if i == j else -1.0
            loss -= math.log(S.expit(z * logits[i][j]))
        if max(range(n), key=lambda j: logits[i][j]) == i:
            hit += 1
    return RichResult(payload={
        "estimate": loss / n, "logits": logits, "acc": hit / n, "n": n,
        "method": "SigLIP pairwise sigmoid loss"})


siglippairwise = siglip_pairwise


def cheatsheet():
    return "siglip: SigLIP pairwise sigmoid loss."
