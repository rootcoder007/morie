# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image-Text Contrastive loss (symmetric InfoNCE over a batch)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_image_text_contrastive"]


def _row_ce(logits):
    """Cross-entropy of each row against its own index (the matching
    pair), via log-sum-exp."""
    m = logits.max(axis=1, keepdims=True)
    logZ = m.ravel() + np.log(np.exp(logits - m).sum(axis=1))
    diag = np.diag(logits)
    return logZ - diag


def kamath_image_text_contrastive(I_emb, T_emb, tau):
    """L_ITC = 0.5 * (CE(sim(I,T)/tau, diag) + CE(sim(T,I)/tau, diag)).

    ``sim`` is cosine similarity, so the temperature has its usual
    meaning; the batch diagonal is the positive pair. A batch of one
    has log-sum-exp over a single logit and therefore loss 0 -- that
    is arithmetic, not learning, so it is refused.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 9,
    image-text contrastive; the named section is not in the 2024 PDF,
    so the loss is implemented exactly as the spec line states
    (InfoNCE, Radford et al. 2021).

    Examples
    --------
    >>> import math
    >>> out = kamath_image_text_contrastive(
    ...     [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], 1.0)
    >>> abs(out["estimate"] - math.log(1 + math.exp(-1.0))) < 1e-12
    True
    >>> out["n"]
    2
    """
    I = np.atleast_2d(np.asarray(I_emb, dtype=float))
    T = np.atleast_2d(np.asarray(T_emb, dtype=float))
    tau = float(tau)
    if I.shape != T.shape:
        raise ValueError(
            f"the two towers must produce matching batches; got "
            f"{I.shape} and {T.shape}.")
    B = I.shape[0]
    if B < 2:
        raise ValueError(
            "InfoNCE needs at least two pairs in the batch; with one "
            "pair there are no negatives and the loss is 0 by "
            "construction.")
    if tau <= 0:
        raise ValueError(f"tau must be positive; got {tau}.")
    ni = np.linalg.norm(I, axis=1)
    nt = np.linalg.norm(T, axis=1)
    if np.any(ni == 0) or np.any(nt == 0):
        raise ValueError(
            "a zero embedding has no direction; cosine similarity is "
            "undefined.")
    S = (I / ni[:, None]) @ (T / nt[:, None]).T
    logits = S / tau
    i2t = _row_ce(logits)
    t2i = _row_ce(logits.T)
    loss = 0.5 * float(i2t.mean() + t2i.mean())
    return RichResult(payload={
        "estimate": loss, "loss": loss,
        "loss_i2t": float(i2t.mean()), "loss_t2i": float(t2i.mean()),
        "similarity": [[float(v) for v in row] for row in S],
        "tau": tau, "n": B,
        "method": "Image-text contrastive (symmetric InfoNCE)"})


def cheatsheet():
    return "kmitc: symmetric InfoNCE over cosine(I,T)/tau, diagonal = positives"
