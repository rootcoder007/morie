# morie.fn -- function file (rootcoder007/morie)
"""Causal LM next-token cross-entropy loss (GPT-style)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_causal_lm_loss"]


def kamath_causal_lm_loss(logits, targets, ignore_index=-100):
    r"""Next-token cross-entropy for an autoregressive language model.

    .. math:: \mathcal{L} = -\frac{1}{T}\sum_{t} \log p(x_t \mid x_{<t}),
              \qquad p = \mathrm{softmax}(\text{logits}_t).

    Computed with the log-sum-exp shift for numerical stability.
    Perplexity is :math:`e^{\mathcal{L}}`, and the loss of a uniform
    model over V tokens is :math:`\ln V` -- the reference point a
    trained model must beat.

    Parameters
    ----------
    logits : array-like, shape (T, V) or (B, T, V)
        Unnormalised scores, already shifted so that row t predicts
        ``targets[t]``.
    targets : array-like of int, shape (T,) or (B, T)
        Gold next-token ids; entries equal to ``ignore_index`` are
        skipped (padding).
    ignore_index : int, default -100
        Target value marking positions to skip.

    Returns
    -------
    RichResult
        keys: ``loss`` (mean over counted positions), ``perplexity``,
        ``token_losses`` (per counted position), ``n_tokens``,
        ``vocab_size``, ``method``.

    References
    ----------
    Kamath, U., Graham, K. L. & Emara, W. (2022). *Transformers for
    Machine Learning: A Deep Dive*. Chapman & Hall/CRC. Ch. 3
    (autoregressive / causal language-model objective).
    """
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets)
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
    if logits.ndim != 2:
        raise ValueError("logits must be (T, V) or (B, T, V).")
    if targets.ndim != 1 or targets.size != logits.shape[0]:
        raise ValueError(f"targets must have {logits.shape[0]} entries, got {targets.size}.")
    V = logits.shape[1]

    keep = targets != ignore_index
    if not keep.any():
        raise ValueError("every target is ignore_index; nothing to score.")
    tgt = targets[keep].astype(int)
    if np.any((tgt < 0) | (tgt >= V)):
        raise ValueError(f"target ids must lie in [0, {V - 1}].")
    lg = logits[keep]

    m = lg.max(axis=1, keepdims=True)
    logZ = m.ravel() + np.log(np.exp(lg - m).sum(axis=1))
    tok = logZ - lg[np.arange(tgt.size), tgt]
    loss = float(tok.mean())

    return RichResult(
        payload={
            "loss": loss,
            "perplexity": float(np.exp(loss)),
            "token_losses": tok,
            "n_tokens": int(tgt.size),
            "vocab_size": int(V),
            "method": "Causal LM cross-entropy (log-sum-exp stabilised)",
        }
    )


def cheatsheet():
    return "kmclm: mean -log p(x_t | x_<t); perplexity = exp(loss); uniform baseline ln V"
