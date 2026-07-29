# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""InfoNCE contrastive loss: pulls positives together, pushes negatives apart."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_contrastive_infonce"]

_METHOD = "InfoNCE contrastive loss"


def _l2norm(A):
    n = np.linalg.norm(A, axis=-1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("cannot cosine-normalise a zero vector.")
    return A / n


def geron_contrastive_infonce(anchors, positives, negatives, tau=0.1, normalize=True):
    r"""InfoNCE loss with one positive and ``N`` negatives per anchor.

    .. math::
        L = -\log \frac{\exp(\mathrm{sim}(a,p)/\tau)}
        {\exp(\mathrm{sim}(a,p)/\tau) + \sum_n \exp(\mathrm{sim}(a,n)/\tau)}

    This is cross-entropy over ``N+1`` candidates with the positive as
    the label, so the loss floors at 0 and its chance value is
    :math:`\log(N+1)`.  The temperature controls how much the hardest
    negatives dominate: small :math:`\tau` sharpens the softmax and makes
    the loss almost entirely about the single closest negative.

    Parameters
    ----------
    anchors : array-like, shape (B, d)
    positives : array-like, shape (B, d)
        One positive per anchor, aligned by row.
    negatives : array-like, shape (N, d) or (B, N, d)
        Negatives shared across the batch, or per anchor.
    tau : float, optional
        Temperature, positive. Default 0.1.
    normalize : bool, optional
        Use cosine similarity (default) rather than raw dot products.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``per_anchor_loss``, ``pos_sim``,
        ``neg_sim``, ``hardest_negative``, ``accuracy`` (positive ranked
        first), ``chance_loss``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 16, Contrastive Learning section.

    Examples
    --------
    One anchor, an identical positive and an orthogonal negative, at
    ``tau = 1``: the loss is ``log(1 + e^-1)``.

    >>> r = geron_contrastive_infonce([[1.0, 0.0]], [[1.0, 0.0]],
    ...                               [[0.0, 1.0]], tau=1.0)
    >>> round(r["loss"], 6)
    0.313262
    >>> r["pos_sim"], r["neg_sim"]
    ([1.0], [[0.0]])
    >>> r["accuracy"]
    1.0

    A negative identical to the positive gives the chance loss ``log 2``:

    >>> import math
    >>> r2 = geron_contrastive_infonce([[1.0, 0.0]], [[1.0, 0.0]],
    ...                                [[1.0, 0.0]], tau=1.0)
    >>> round(r2["loss"], 6) == round(math.log(2), 6)
    True
    """
    A = np.atleast_2d(np.asarray(anchors, dtype=float))
    P = np.atleast_2d(np.asarray(positives, dtype=float))
    N = np.asarray(negatives, dtype=float)
    if A.shape != P.shape:
        raise ValueError(
            f"anchors and positives must have the same shape, got {A.shape} and {P.shape}."
        )
    if A.size == 0:
        raise ValueError("anchors is empty.")
    B, d = A.shape
    if N.ndim == 2:
        if N.shape[1] != d:
            raise ValueError(f"negatives width {N.shape[1]} != anchor width {d}.")
        N = np.broadcast_to(N[None, :, :], (B, N.shape[0], d))
    elif N.ndim == 3:
        if N.shape[0] != B or N.shape[2] != d:
            raise ValueError(
                f"per-anchor negatives must have shape (B, N, d) = ({B}, N, {d}), got {N.shape}."
            )
    else:
        raise ValueError(f"negatives must be 2-D or 3-D, got ndim={N.ndim}.")
    if N.shape[1] == 0:
        raise ValueError("at least one negative per anchor is required.")
    for name, arr in (("anchors", A), ("positives", P), ("negatives", N)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError(f"tau must be a positive finite float, got {tau}.")

    if normalize:
        A = _l2norm(A)
        P = _l2norm(P)
        N = _l2norm(N)

    pos = np.sum(A * P, axis=1)                       # (B,)
    neg = np.einsum("bd,bnd->bn", A, N)               # (B, n)
    logits = np.concatenate([pos[:, None], neg], axis=1) / tau
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    per_anchor = lse - logits[:, 0]
    loss = float(per_anchor.mean())
    acc = float(np.mean(logits.argmax(axis=1) == 0))

    return RichResult(
        title="InfoNCE contrastive loss",
        summary_lines=[("Loss", loss), ("Negatives per anchor", int(N.shape[1]))],
        interpretation=(
            f"Chance loss is log(1+N) = {float(np.log(1 + N.shape[1])):.4f}."
        ),
        payload={
            "loss": loss,
            "per_anchor_loss": per_anchor.tolist(),
            "pos_sim": pos.tolist(),
            "neg_sim": neg.tolist(),
            "hardest_negative": neg.max(axis=1).tolist(),
            "accuracy": acc,
            "chance_loss": float(np.log(1 + N.shape[1])),
            "tau": tau,
            "estimate": loss,
            "n": int(B),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grctr: InfoNCE -- cross-entropy over 1 positive and N negatives at temperature tau"
