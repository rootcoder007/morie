# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Supervised fine-tuning (SFT) cross-entropy on (prompt, response) pairs."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_sft_objective"]

_METHOD = "SFT masked cross-entropy objective"


def _log_softmax(Z):
    M = Z.max(axis=-1, keepdims=True)
    Z = Z - M
    return Z - np.log(np.exp(Z).sum(axis=-1, keepdims=True))


def geron_sft_objective(logits, response_mask, targets):
    r"""Cross-entropy over the response tokens only.

    .. math::
        L_{\text{SFT}} = -\frac{1}{|R|}\sum_{t \in R}
            \log p_{\theta}\bigl(r_t \mid \text{prompt}, r_{<t}\bigr)

    The mask is the entire difference between SFT and ordinary language
    modelling.  Train on the prompt tokens too and the model learns to
    *generate questions*, which is not what an assistant is for; and the
    normaliser is :math:`|R|`, the number of *unmasked* tokens, so a long
    prompt does not dilute the loss.  Both are enforced: an all-zero mask
    raises rather than dividing by zero.

    Parameters
    ----------
    logits : array-like, shape (T, V)
        Next-token logits.
    response_mask : array-like of bool, shape (T,)
        True where the token belongs to the response.
    targets : array-like of int, shape (T,)

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``per_token``, ``perplexity``,
        ``n_response_tokens``, ``token_logprobs``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 15, Supervised Fine-Tuning section.

    Examples
    --------
    Two tokens, only the second is a response token.  Uniform logits over
    2 classes cost ``log 2 = 0.693147`` and the confident prompt token is
    ignored entirely:

    >>> logits = [[10.0, 0.0], [0.0, 0.0]]
    >>> r = geron_sft_objective(logits, [False, True], [0, 1])
    >>> round(r["loss"], 6)
    0.693147
    >>> r["n_response_tokens"]
    1

    Masking everything is a caller error, not a zero loss:

    >>> geron_sft_objective(logits, [False, False], [0, 1])
    Traceback (most recent call last):
        ...
    ValueError: response_mask selects no tokens; there is nothing to fine-tune on.
    """
    Z = np.atleast_2d(np.asarray(logits, dtype=float))
    if Z.ndim != 2 or Z.size == 0:
        raise ValueError(f"logits must be a non-empty (T, V) matrix, got shape {Z.shape}.")
    if not np.all(np.isfinite(Z)):
        raise ValueError("logits contains non-finite values.")
    T, V = Z.shape
    tgt = np.asarray(targets).ravel()
    if tgt.size != T:
        raise ValueError(f"targets has {tgt.size} entries but logits has {T} positions.")
    if not np.all(tgt == np.round(np.asarray(tgt, dtype=float))):
        raise ValueError("targets must be integer token ids.")
    tgt = tgt.astype(int)
    if tgt.min() < 0 or tgt.max() >= V:
        raise ValueError(f"target ids must lie in [0, {V - 1}].")
    mask = np.asarray(response_mask).ravel()
    if mask.size != T:
        raise ValueError(f"response_mask has {mask.size} entries but logits has {T} positions.")
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("response_mask selects no tokens; there is nothing to fine-tune on.")

    logp = _log_softmax(Z)
    tok = logp[np.arange(T), tgt]
    per = -tok[mask]
    loss = float(per.mean())

    return RichResult(
        title="SFT objective",
        summary_lines=[("Loss", loss), ("Response tokens", int(mask.sum())),
                       ("Perplexity", float(np.exp(loss)))],
        payload={
            "loss": loss,
            "per_token": per.tolist(),
            "perplexity": float(np.exp(loss)),
            "n_response_tokens": int(mask.sum()),
            "token_logprobs": tok.tolist(),
            "estimate": loss,
            "n": int(T),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grsft: -mean log p over MASKED response tokens only; prompt tokens excluded from the average"
