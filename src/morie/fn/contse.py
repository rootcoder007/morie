# morie.fn -- function file (rootcoder007/morie)
"""SimCSE contrastive sentence objective."""

import math

from . import _s03core as core
from ._richresult import RichResult
from .clipsi import l2_normalize

__all__ = ["contrastive_sent"]


def contrastive_sent(sentences, tau=0.05, dropout=0.1, seed=42):
    """
    SimCSE contrastive sentence embedding

    Formula: InfoNCE on dropout-augmented pairs

    l_i = -log exp(sim(h_i, h_i+)/tau) / sum_j exp(sim(h_i, h_j+)/tau).
    The positive is the SAME sentence encoded twice under different
    dropout masks, so the objective needs no labelled pairs at all.
    When every similarity is identical the loss is exactly log N, which
    is the value it must return on a degenerate batch.

    Parameters
    ----------
    sentences : array-like
        n x d matrix of sentence embeddings before augmentation.
    tau : float
        Temperature, strictly positive.
    dropout : float
        Dropout rate in [0, 1) used to build the two views.
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (InfoNCE loss), loss, per_item, alignment,
        uniformity, n, d.

    References
    ----------
    Gao, Yao & Chen (2021), SimCSE: Simple Contrastive Learning of
    Sentence Embeddings, EMNLP 2021:6894-6910.
    """
    from . import _array_core as np
    H = core.mat(sentences)
    n = len(H)
    if n == 0:
        raise ValueError("empty input: sentences has no rows")
    d = len(H[0])
    if not (tau > 0.0):
        raise ValueError("tau must be strictly positive")
    if not (0.0 <= dropout < 1.0):
        raise ValueError("dropout must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    keep = 1.0 - dropout
    A, B = [], []
    for i in range(n):
        ra, rb = [], []
        for k in range(d):
            ma = 0.0 if float(rng.uniform(0.0, 1.0)) < dropout else 1.0 / keep
            mb = 0.0 if float(rng.uniform(0.0, 1.0)) < dropout else 1.0 / keep
            ra.append(H[i][k] * ma)
            rb.append(H[i][k] * mb)
        A.append(l2_normalize(ra))
        B.append(l2_normalize(rb))
    per = []
    for i in range(n):
        s = [sum(A[i][k] * B[j][k] for k in range(d)) / tau for j in range(n)]
        mx = max(s)
        lse = mx + math.log(sum(math.exp(v - mx) for v in s))
        per.append(lse - s[i])
    loss = sum(per) / n
    align = sum(sum((A[i][k] - B[i][k]) ** 2 for k in range(d))
                for i in range(n)) / n
    unif = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            unif += math.exp(-2.0 * sum((A[i][k] - A[j][k]) ** 2
                                        for k in range(d)))
            cnt += 1
    unif = math.log(unif / cnt) if cnt else float("nan")
    return RichResult(payload={
        "estimate": loss,
        "loss": loss,
        "per_item": per,
        "alignment": align,
        "uniformity": unif,
        "n": n,
        "d": d,
        "method": "SimCSE contrastive sentence objective",
    })


def cheatsheet():
    return "contse: SimCSE contrastive sentence objective"


# compact alias per ledger/NAMING.md
contrastivesent = contrastive_sent
